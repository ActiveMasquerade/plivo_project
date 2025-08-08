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

    const buttonStyle = {
        padding: "0.75rem 1.5rem",
        border: "none",
        borderRadius: "6px",
        fontWeight: "bold",
        fontSize: "1rem",
        color: "#fff",
        cursor: "pointer",
        transition: "background 0.3s ease",
        margin: "0.5rem"
    };

    return (
        <div style={{
            maxWidth: "1000px",
            margin: "2rem auto",
            padding: "2rem",
            backgroundColor: "#fff",
            borderRadius: "12px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.1)",
            textAlign: "center"
        }}>
            <h2 style={{ marginBottom: "1.5rem" }}>Image Captioning</h2>

            <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                style={{ display: "none" }}
                onChange={handleFileChange}
            />

            <button
                onClick={handleUploadClick}
                style={{ ...buttonStyle, backgroundColor: "#28a745" }}
                onMouseOver={(e) => e.target.style.backgroundColor = "#218838"}
                onMouseOut={(e) => e.target.style.backgroundColor = "#28a745"}
            >
                Choose Image
            </button>

            {preview && (
                <div style={{
                    margin: "1.5rem 0",
                    display: "flex",
                    justifyContent: "center"
                }}>
                    <img
                        src={preview}
                        alt="Preview"
                        style={{
                            maxWidth: "100%",
                            maxHeight: "400px",
                            borderRadius: "8px",
                            objectFit: "contain",
                            boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                        }}
                    />
                </div>
            )}

            {file && (
                <button
                    onClick={handleCaptionRequest}
                    disabled={loading}
                    style={{
                        ...buttonStyle,
                        backgroundColor: loading ? "#6c757d" : "#007bff",
                        cursor: loading ? "not-allowed" : "pointer"
                    }}
                    onMouseOver={(e) => !loading && (e.target.style.backgroundColor = "#0056b3")}
                    onMouseOut={(e) => !loading && (e.target.style.backgroundColor = "#007bff")}
                >
                    {loading ? "Processing..." : "Get Description"}
                </button>
            )}

            {description && (
                <p style={{
                    marginTop: "1.5rem",
                    fontSize: "1.1rem",
                    background: "#f8f9fa",
                    padding: "1rem",
                    borderRadius: "8px",
                    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.05)"
                }}>
                    <strong>Description:</strong> {description}
                </p>
            )}

            <div style={{
                display: "flex",
                justifyContent: "center",
                marginTop: "2rem",
                gap: "1rem"
            }}>
                <button
                    style={{ ...buttonStyle, backgroundColor: "#17a2b8" }}
                    onMouseOver={(e) => e.target.style.backgroundColor = "#117a8b"}
                    onMouseOut={(e) => e.target.style.backgroundColor = "#17a2b8"}
                >
                    Get Speech Diarization
                </button>

                <button
                    style={{ ...buttonStyle, backgroundColor: "#ffc107", color: "#000" }}
                    onMouseOver={(e) => e.target.style.backgroundColor = "#e0a800"}
                    onMouseOut={(e) => e.target.style.backgroundColor = "#ffc107"}
                >
                    Summarize Document
                </button>
            </div>
        </div>
    );
}

export default App;
