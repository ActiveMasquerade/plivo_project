import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'

function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [description, setDescription] = useState("");
    const [loading, setLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleCaptionRequest = async () => {
        if (!file) return;
        setLoading(true);
        setDescription("");

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post(
                "https://sadaman-plivo-image.hf.space/caption", // FastAPI endpoint
                formData,
                { headers: { "Content-Type": "multipart/form-data" } }
            );
            setDescription(res.data.description);
        } catch (err) {
            console.error(err);
            setDescription("Error: Could not get description");
        }
        setLoading(false);
    };

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
        }
    };

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    return (
        <>
            <div style={{ maxWidth: "400px", margin: "2rem auto", textAlign: "center" }}>
                <h2>Image Captioning</h2>

                {/* Hidden File Input */}
                <input
                    type="file"
                    accept="image/*"
                    ref={fileInputRef}
                    style={{ display: "none" }}
                    onChange={handleFileChange}
                />

                {/* Upload Button */}
                <button
                    onClick={handleUploadClick}
                    style={{
                        padding: "0.5rem 1rem",
                        border: "none",
                        borderRadius: "4px",
                        backgroundColor: "#28a745",
                        color: "#fff",
                        cursor: "pointer",
                        fontWeight: "bold",
                    }}
                >
                    Choose Image
                </button>

                {/* Image Preview */}
                {preview && (
                    <div style={{ margin: "1rem 0" }}>
                        <img
                            src={preview}
                            alt="Preview"
                            style={{ maxWidth: "100%", borderRadius: "8px" }}
                        />
                    </div>
                )}

                {/* Get Description Button */}
                {file && (
                    <button
                        onClick={handleCaptionRequest}
                        disabled={loading}
                        style={{
                            marginTop: "1rem",
                            padding: "0.5rem 1rem",
                            border: "none",
                            borderRadius: "4px",
                            backgroundColor: loading ? "#6c757d" : "#007bff",
                            color: "#fff",
                            cursor: loading ? "not-allowed" : "pointer",
                        }}
                    >
                        {loading ? "Processing..." : "Get Description"}
                    </button>
                )}

                {/* Description Output */}
                {description && (
                    <p style={{ marginTop: "1rem" }}>
                        <strong>Description:</strong> {description}
                    </p>
                )}
            </div>

            {/* Placeholder buttons for other features */}
            <button>
                Get Speech diarization
            </button>
            <button>
                Summarize Document
            </button>
        </>
    );
}

export default App;
