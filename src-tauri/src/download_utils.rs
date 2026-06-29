use anyhow::{Context, Result};
use reqwest::Client;
use std::path::Path;
use tauri::Emitter;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt; // Required for file.write_all and file.flush

#[derive(Clone, serde::Serialize)]
pub struct DownloadProgressPayload {
    pub model_name: String,
    pub downloaded: u64,
    pub total: u64,
}

/// Streams a file from a URL to disk via a shared reqwest Client.
///
/// Uses chunked streaming to keep memory usage low, regardless of file size.
/// Employs an atomic write strategy (writing to a .tmp file and renaming) 
/// to prevent corrupted files, and cleans up partial downloads on failure.
/// Emits progress events to the Tauri frontend.
pub async fn download_file(
    client: &Client,
    url: &str,
    dest: &Path,
    app_handle: &tauri::AppHandle,
    model_name: &str,
) -> Result<()> {
    log::info!(
        "Starting streaming download from {} to {}",
        url,
        dest.display()
    );

    let mut response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to connect to {}", url))?
        .error_for_status()
        .with_context(|| format!("Server returned an error for {}", url))?;

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let tmp_dest = dest.with_extension("tmp");
    let mut file = File::create(&tmp_dest)
        .await
        .with_context(|| format!("Failed to create temporary file {:?}", tmp_dest))?;

    // Isolate the streaming process in an async block so we can catch errors
    // and clean up the temporary file before returning.
    let download_result: Result<()> = async {
        let mut last_emit_time = std::time::Instant::now();
        // Stream the response body in chunks
        while let Some(chunk) = response
            .chunk()
            .await
            .with_context(|| "Error while reading the network stream")?
        {
            // Write each chunk directly to disk as it arrives
            file.write_all(&chunk)
                .await
                .with_context(|| "Failed to write data chunk to disk")?;
            
            downloaded += chunk.len() as u64;

            // Throttle progress events to avoid flooding the UI (e.g., every 100ms)
            if last_emit_time.elapsed().as_millis() > 100 {
                let _ = app_handle.emit(
                    "ai-model-download-progress",
                    DownloadProgressPayload {
                        model_name: model_name.to_string(),
                        downloaded,
                        total: total_size,
                    },
                );
                last_emit_time = std::time::Instant::now();
            }
        }

        // Ensure all remaining data in the OS buffer is physically written to the drive
        file.flush()
            .await
            .with_context(|| "Failed to flush file to disk")?;

        Ok(())
    }
    .await;

    // Explicitly drop the file handle. This is crucial for Windows, 
    // which will block the rename operation if the file is still open.
    drop(file);

    // Error handling: Clean up the half-downloaded file and bubble up the error
    if let Err(ref e) = download_result {
        log::error!("Download interrupted ({}). Cleaning up temporary file...", e);
        let _ = fs::remove_file(&tmp_dest).await; // Ignore errors during cleanup
        return download_result; // Return the exact error
    }

    // Atomically rename the complete temporary file to the final destination
    fs::rename(&tmp_dest, dest)
        .await
        .with_context(|| format!("Failed to rename {:?} to {:?}", tmp_dest, dest))?;

    log::info!("Download completed successfully to {}", dest.display());
    Ok(())
}
