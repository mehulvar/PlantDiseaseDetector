export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SOURCE_BYTES = 40 * 1024 * 1024;
const MAX_UPLOAD_BYTES = 1.5 * 1024 * 1024;
const MAX_DIMENSION = 1600;
const JPEG_QUALITY = 0.9;

export class ImageInputError extends Error {}

/**
 * Validate a user-chosen photo and, when it is large, downscale it in the browser so it
 * uploads quickly and stays under the server limit. The model only needs 224 px.
 */
export async function prepareImage(file: File): Promise<Blob> {
  if (file.type && !ACCEPTED_TYPES.includes(file.type)) {
    throw new ImageInputError("Please choose a JPEG, PNG or WebP photo.");
  }
  if (file.size > MAX_SOURCE_BYTES) {
    throw new ImageInputError("That file is too large. Please choose a photo under 40 MB.");
  }
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new ImageInputError("This file could not be opened as an image. Try a JPEG or PNG photo.");
  }
  try {
    const longest = Math.max(bitmap.width, bitmap.height);
    if (file.size <= MAX_UPLOAD_BYTES && longest <= MAX_DIMENSION) return file;
    const scale = Math.min(1, MAX_DIMENSION / longest);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const context = canvas.getContext("2d");
    if (!context) throw new ImageInputError("Your browser could not process this image.");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
    if (!blob) throw new ImageInputError("Your browser could not process this image.");
    return blob;
  } finally {
    bitmap.close();
  }
}
