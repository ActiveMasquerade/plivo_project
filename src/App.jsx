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
                "https://sadaman-plivo-image.hf.space/caption",
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
        <div style={{
            minHeight: "100vh",
            backgroundColor: "#f5f6fa",
            padding: "2rem",
            display: "flex",
            justifyContent: "center",
            alignItems: "center"
        }}>
            <div style={{
                backgroundColor: "#fff",
                padding: "2rem",
                borderRadius: "10px",
                boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
                maxWidth: "500px",
                width: "100%",
                textAlign: "center"
            }}>
                <h2 style={{ marginBottom: "1rem", color: "#333" }}>Image Captioning</h2>

                <input
                    type="file"
                    accept="image/*"
                    ref={fileInputRef}
                    style={{ display: "none" }}
                    onChange={handleFileChange}
                />

                <button
                    onClick={handleUploadClick}
                    style={{
                        padding: "0.5rem 1rem",
                        border: "none",
                        borderRadius: "6px",
                        backgroundColor: "#28a745",
                        color: "#fff",
                        cursor: "pointer",
                        fontWeight: "bold",
                        marginBottom: "1rem"
                    }}
                >
                    Choose Image
                </button>

                {preview && (
                    <div style={{
                        margin: "1rem 0",
                        border: "1px solid #ddd",
                        borderRadius: "8px",
                        overflow: "hidden"
                    }}>
                        <img
                            src={preview}
                            alt="Preview"
                            style={{ maxWidth: "100%", display: "block" }}
                        />
                    </div>
                )}

                {file && (
                    <button
                        onClick={handleCaptionRequest}
                        disabled={loading}
                        style={{
                            marginTop: "0.5rem",
                            padding: "0.5rem 1rem",
                            border: "none",
                            borderRadius: "6px",
                            backgroundColor: loading ? "#6c757d" : "#007bff",
                            color: "#fff",
                            cursor: loading ? "not-allowed" : "pointer",
                            fontWeight: "bold"
                        }}
                    >
                        {loading ? "Processing..." : "Get Description"}
                    </button>
                )}

                {description && (
                    <div style={{
                        marginTop: "1rem",
                        padding: "0.8rem",
                        border: "1px solid #ddd",
                        borderRadius: "6px",
                        backgroundColor: "#f9f9f9",
                        textAlign: "left"
                    }}>
                        <strong>Description:</strong> {description}
                    </div>
                )}

                <div style={{ marginTop: "2rem" }}>
                    <h3 style={{ color: "#333", marginBottom: "0.8rem" }}>Other Features</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                        <button style={{
                            padding: "0.5rem 1rem",
                            border: "none",
                            borderRadius: "6px",
                            backgroundColor: "#17a2b8",
                            color: "#fff",
                            cursor: "pointer",
                            fontWeight: "bold"
                        }}>
                            Get Speech Diarization
                        </button>
                        <button style={{
                            padding: "0.5rem 1rem",
                            border: "none",
                            borderRadius: "6px",
                            backgroundColor: "#ffc107",
                            color: "#333",
                            cursor: "pointer",
                            fontWeight: "bold"
                        }}>
                            Summarize Document
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
