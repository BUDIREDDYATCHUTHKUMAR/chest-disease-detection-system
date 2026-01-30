import React, { useRef, useState } from 'react';
import { Upload, X, FileImage, ScanLine, Smartphone } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const UploadZone = ({ files, setFiles, onUpload, scanning }) => {
    const fileInputRef = useRef(null);
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFiles(e.dataTransfer.files);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFiles(e.target.files);
        }
    };

    const handleFiles = (newFiles) => {
        // Convert FileList to Array and add preview URLs
        const filesArray = Array.from(newFiles).map(file => ({
            file,
            preview: URL.createObjectURL(file),
            id: Math.random().toString(36).substr(2, 9)
        }));
        setFiles(prev => [...prev, ...filesArray]);
    };

    const removeFile = (id) => {
        setFiles(prev => prev.filter(f => f.id !== id));
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            {/* Main Upload Area */}
            <motion.div
                layout
                className={twMerge(
                    "relative border-2 border-dashed rounded-2xl p-10 transition-all duration-300 text-center group cursor-pointer overflow-hidden",
                    dragActive ? "border-accent bg-accent/5 scale-[1.01]" : "border-slate-700 bg-surface hover:border-slate-500",
                    files.length > 0 ? "h-auto" : "h-[400px] flex flex-col items-center justify-center"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={handleChange}
                    accept="image/*"
                />

                {/* Dynamic Content based on State */}
                <AnimatePresence mode="wait">
                    {files.length === 0 ? (
                        <motion.div
                            key="empty"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-6"
                        >
                            <div className="w-24 h-24 rounded-full bg-slate-800/50 flex items-center justify-center relative group-hover:bg-slate-800 transition-colors">
                                <div className="absolute inset-0 rounded-full border border-accent/20 animate-pulse-slow"></div>
                                <Upload className="w-10 h-10 text-slate-400 group-hover:text-accent transition-colors" />
                            </div>
                            <div>
                                <h3 className="text-xl font-medium text-white mb-2">Upload Chest X-ray</h3>
                                <p className="text-slate-500 text-sm max-w-xs mx-auto">
                                    Drag and drop DICOM, PNG, or JPG files here (PA / AP View supported)
                                </p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="preview"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="w-full"
                        >
                            {/* Selected Image Preview (Last one uploaded usually focus) */}
                            <div className="relative w-full aspect-[16/9] md:aspect-[2/1] bg-black/40 rounded-lg overflow-hidden mb-8 border border-slate-700 group-hover:border-slate-600 transition-colors">
                                <img
                                    src={files[files.length - 1].preview}
                                    alt="Preview"
                                    className="w-full h-full object-contain"
                                />

                                {/* Scanning Animation Overlay */}
                                {scanning && (
                                    <motion.div
                                        initial={{ top: "0%" }}
                                        animate={{ top: "100%" }}
                                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                        className="absolute left-0 right-0 h-1 bg-accent shadow-[0_0_20px_#06b6d4] z-20"
                                    />
                                )}
                                {scanning && (
                                    <div className="absolute inset-0 bg-accent/5 z-10 animate-pulse"></div>
                                )}

                                <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded-full border border-white/10 flex items-center gap-2">
                                    <FileImage className="w-3 h-3 text-accent" />
                                    <span className="text-xs font-mono text-slate-300">{files[files.length - 1].file.name}</span>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Scanning Decor elements */}
                <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-accent/20 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-accent/20 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-accent/20 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-accent/20 rounded-br-lg"></div>
            </motion.div>

            {/* Multi-file Horizontal Scroll */}
            {files.length > 0 && (
                <div className="mt-6 flex gap-4 overflow-x-auto pb-4 custom-scrollbar">
                    {files.map((f, idx) => (
                        <div key={f.id} className="relative flex-shrink-0 w-24 h-24 bg-surface rounded-lg border border-slate-700 overflow-hidden group/thumb">
                            <img src={f.preview} className="w-full h-full object-cover" />
                            <button
                                onClick={(e) => { e.stopPropagation(); removeFile(f.id); }}
                                className="absolute top-1 right-1 p-1 bg-red-500/80 rounded-full opacity-0 group-hover/thumb:opacity-100 transition-opacity"
                            >
                                <X className="w-3 h-3 text-white" />
                            </button>
                            {idx === files.length - 1 && (
                                <div className="absolute inset-0 border-2 border-accent pointer-events-none rounded-lg"></div>
                            )}
                        </div>
                    ))}
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="flex-shrink-0 w-24 h-24 flex items-center justify-center border border-dashed border-slate-700 rounded-lg hover:border-slate-500 hover:bg-white/5 transition-all text-slate-500"
                    >
                        <Upload className="w-5 h-5" />
                    </button>
                </div>
            )}

            {/* Analyze Button */}
            {files.length > 0 && !scanning && (
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={onUpload}
                    className="w-full mt-8 py-4 bg-gradient-to-r from-accent to-medical text-primary font-bold text-lg rounded-xl shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] transition-all flex items-center justify-center gap-3"
                >
                    <ScanLine className="w-5 h-5" />
                    ANALYZE X-RAY
                </motion.button>
            )}

            {scanning && (
                <div className="w-full mt-8 py-4 bg-surface border border-accent/20 rounded-xl flex items-center justify-center gap-3 text-accent font-mono animate-pulse">
                    <ScanLine className="w-5 h-5 animate-spin" />
                    AI MODEL SCANNING LUNGS...
                </div>
            )}
        </div>
    );
};

export default UploadZone;
