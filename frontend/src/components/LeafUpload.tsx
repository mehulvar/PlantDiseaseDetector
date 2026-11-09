import { useState, type DragEvent } from "react";
import type { Crop } from "../api/types";
import { ACCEPTED_TYPES } from "../lib/image";
import { UploadIcon } from "./Icons";

export type CropChoice = Crop | "auto";

interface LeafUploadProps {
  previewUrl: string | null;
  fileName: string | null;
  crop: CropChoice;
  crops: readonly Crop[];
  onFile: (file: File) => void;
  onClear: () => void;
  onCrop: (crop: CropChoice) => void;
}

export default function LeafUpload({ previewUrl, fileName, crop, crops, onFile, onClear, onCrop }: LeafUploadProps) {
  const [dragging, setDragging] = useState(false);

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div className="card">
      <div className="card__head">
        <h3 className="card__title display">Leaf photo</h3>
        <span className="mono bracket">Optional</span>
      </div>

      <label
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          aria-label="Upload a leaf photo"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
        {previewUrl ? (
          <img className="dropzone__preview" src={previewUrl} alt="Selected leaf" />
        ) : (
          <div className="dropzone__empty">
            <UploadIcon />
            <strong>Drop a leaf photo, or tap to choose</strong>
            <p className="mono">JPEG · PNG · WebP</p>
          </div>
        )}
      </label>

      {previewUrl && (
        <div className="leaf__meta">
          <span className="leaf__file mono">{fileName ?? "Photo"}</span>
          <button type="button" className="ghost-btn" onClick={onClear}>
            Remove photo
          </button>
        </div>
      )}

      <fieldset className="crop-picker">
        <legend className="mono">Crop</legend>
        <div className="segmented">
          {(["auto", ...crops] as CropChoice[]).map((option) => (
            <label key={option}>
              <input
                type="radio"
                name="crop"
                value={option}
                checked={crop === option}
                onChange={() => onCrop(option)}
              />
              <span>{option === "auto" ? "Detect" : option}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <p className="field-hint">
        Photograph one leaf, filling the frame, in even light. Supported: bell pepper, potato and
        tomato. Choosing the crop narrows the diagnosis.
      </p>
    </div>
  );
}
