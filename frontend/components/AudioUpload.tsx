"use client";

import React, { useState } from "react";

export default function AudioUpload() {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;

    if (selectedFile && selectedFile.type.startsWith("audio/")) {
      setFile(selectedFile);
    } else {
      alert("Please select a valid audio file");
    }
  };

  const uploadAudio = async () => {
    if (!file) return alert("Select a file first");

    const formData = new FormData();
    formData.append("audio", file);

    const res = await fetch("/api/upload-audio", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    alert("Uploaded to: " + data.url);
  };

  return (
    <div>
      <h2>Upload Audio (TSX)</h2>

      <input
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
      />

      {file && <p>Selected: {file.name}</p>}

      <button onClick={uploadAudio}>Upload</button>
    </div>
  );
}
