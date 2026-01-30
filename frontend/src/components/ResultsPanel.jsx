import React from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, FileText, Stethoscope, ChevronRight, Activity, Download, CheckCircle2, Lightbulb } from 'lucide-react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

const RECOMMENDATIONS = {
    'Atelectasis': [
        "Deep breathing exercises (Incentive Spirometry)",
        "Early mobilization and walking",
        "Chest physiotherapy to clear airways",
        "Stay hydrated to thin mucus"
    ],
    'Cardiomegaly': [
        "Strict salt restriction in diet",
        "Fluid intake management",
        "Regular cardiac monitoring",
        "Weight management and light aerobic activity"
    ],
    'Consolidation': [
        "Complete course of prescribed antibiotics (if bacterial)",
        "Rest and adequate hydration",
        "Steam inhalation to loosen mucus",
        "Sleep in an elevated position"
    ],
    'Edema': [
        "Low-sodium diet to reduce fluid retention",
        "Leg elevation while resting",
        "Compliance with diuretic medications",
        "Daily weight monitoring"
    ],
    'Effusion': [
        "High-protein diet to replace lost nutrients",
        "Breathing exercises to re-expand lungs",
        "Report sudden shortness of breath immediately",
        "Regular follow-up X-rays"
    ],
    'Emphysema': [
        "Immediate smoking cessation",
        "Pulmonary rehabilitation exercises",
        "Breathing techniques (pursed-lip breathing)",
        "Avoidance of air pollutants and dust"
    ],
    'Fibrosis': [
        "Oxygen therapy compliance if prescribed",
        "Pulmonary rehabilitation to maintain lung function",
        "Vaccination against flu and pneumonia",
        "Regular monitoring of oxygen levels"
    ],
    'Hernia': [
        "Avoid heavy lifting or straining",
        "Weight management to reduce abdominal pressure",
        "Eat smaller, frequent meals",
        "Surgical consultation may be required"
    ],
    'Infiltration': [
        "Rest and plenty of fluids",
        "Isolate if contagious (viral/bacterial)",
        "Monitor temperature regularly",
        "Deep breathing to prevent secondary infection"
    ],
    'Mass': [
        "Urgent CT scan for characterization",
        "Biopsy may be recommended",
        "Oncology consultation",
        "Do not delay follow-up appointments"
    ],
    'Nodule': [
        "Follow-up CT scan in 3-6 months",
        "Smoking cessation",
        "Monitor for cough or blood in sputum",
        "Previous X-ray comparison"
    ],
    'Pleural_Thickening': [
        "Breathing exercises to maintain chest expansion",
        "Pain management if symptomatic",
        "Occupational history review (asbestos exposure)",
        "Regular pulmonary function tests"
    ],
    'Pneumonia': [
        "Finish entire course of antibiotics",
        "Plenty of rest and fluids",
        "Cool mist humidifier",
        "Fever management"
    ],
    'Pneumothorax': [
        "Avoid air travel or diving until cleared",
        "Stop smoking immediately",
        "Report sudden chest pain or breathlessness",
        "Follow-up X-ray to ensure full expansion"
    ]
};

const ResultsPanel = ({ result }) => {
    const [showHeatmap, setShowHeatmap] = React.useState(false);

    if (!result) return null;

    // Helper to determine color based on probability
    const getRiskLevel = (prob) => {
        if (prob > 0.7) return { color: 'text-red-500', bg: 'bg-red-500', label: 'High Risk' };
        if (prob > 0.3) return { color: 'text-warning', bg: 'bg-warning', label: 'Moderate' };
        return { color: 'text-emerald-400', bg: 'bg-emerald-400', label: 'Low Risk' };
    };

    const topRisk = getRiskLevel(result.top_probability);
    const suggestions = RECOMMENDATIONS[result.top_finding] || ["Consult a specialist for specific advice.", "Monitor symptoms closely."];

    const generatePDF = () => {
        const doc = new jsPDF();

        // Header
        doc.setFillColor(11, 17, 32); // Deep Navy
        doc.rect(0, 0, 210, 40, 'F');
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(22);
        doc.text("AI CHESTCARE DIAGNOSTIC REPORT", 105, 20, { align: "center" });

        // Metadata
        doc.setTextColor(100, 116, 139);
        doc.setFontSize(10);
        doc.text(`Report ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, 15, 50);
        doc.text(`Date: ${new Date().toLocaleString()}`, 15, 55);

        // Primary Findings
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(14);
        doc.text("PRIMARY FINDING", 15, 70);

        doc.setFillColor(240, 249, 255);
        doc.rect(15, 75, 180, 25, 'F');
        doc.setTextColor(6, 182, 212); // Cyan
        doc.setFontSize(18);
        doc.text(`${result.top_finding} (${(result.top_probability * 100).toFixed(1)}%)`, 20, 90);

        // Recommendations Section in PDF
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(14);
        doc.text("RECOMMENDED MEASURES", 15, 115);

        const recoData = suggestions.map(s => [s]);
        autoTable(doc, {
            startY: 120,
            body: recoData,
            theme: 'plain',
            bodyStyles: { fontSize: 11, cellPadding: 2 }
        });

        // Prediction Table
        const tableData = Object.entries(result.prediction)
            .sort(([, a], [, b]) => b - a)
            .map(([label, prob]) => [label, `${(prob * 100).toFixed(2)}%`]);

        doc.text("DIFFERENTIAL DIAGNOSIS", 15, doc.lastAutoTable.finalY + 15);

        autoTable(doc, {
            startY: doc.lastAutoTable.finalY + 20,
            head: [['Condition', 'Probability']],
            body: tableData,
            theme: 'grid',
            headStyles: { fillColor: [6, 182, 212] }
        });

        // Disclaimer
        doc.setFontSize(8);
        doc.setTextColor(150);
        doc.text("This report is generated by an AI model and is not a definitive medical diagnosis.", 105, 280, { align: "center" });

        doc.save(`ChestCare_Report_${new Date().toISOString().slice(0, 10)}.pdf`);
    };

    const handleConsult = () => {
        // Simple mock interaction
        const btn = document.getElementById('consult-btn');
        if (btn) {
            btn.innerHTML = `<span class='flex items-center gap-2'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle-2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> Request Sent</span>`;
            btn.classList.remove('bg-accent/10', 'text-accent');
            btn.classList.add('bg-emerald-500/20', 'text-emerald-500', 'border-emerald-500/30');
            setTimeout(() => {
                alert("Consultation request sent to the on-call radiologist securely.");
            }, 100);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-4xl mx-auto mt-12"
        >
            <div className="grid lg:grid-cols-3 gap-8">
                {/* Main Diagnosis Card */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="glass-panel p-8 rounded-2xl relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-accent to-secondary"></div>

                        <div className="flex items-start justify-between mb-8">
                            <div>
                                <div className="text-xs font-mono text-slate-400 uppercase tracking-widest mb-2">Primary Diagnosis</div>
                                <h2 className="text-4xl font-bold text-white mb-2">{result.top_finding}</h2>
                                <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${topRisk.bg}/10 border ${topRisk.bg}/20`}>
                                    <Activity className={`w-4 h-4 ${topRisk.color}`} />
                                    <span className={`text-xs font-mono font-bold ${topRisk.color}`}>{topRisk.label} Detectd</span>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-5xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent to-white">
                                    {(result.top_probability * 100).toFixed(1)}%
                                </div>
                                <div className="text-xs text-slate-500 mt-1">Confidence Score</div>
                            </div>
                        </div>

                        {/* Recommendations Section */}
                        <div className="mb-8 p-4 bg-white/5 rounded-xl border border-white/10">
                            <h3 className="text-sm font-semibold text-accent mb-3 flex items-center gap-2">
                                <Lightbulb className="w-4 h-4" />
                                Recommended Measures
                            </h3>
                            <ul className="space-y-2">
                                {suggestions.map((item, idx) => (
                                    <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                                        <span className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5 flex-shrink-0"></span>
                                        {item}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Heatmap Toggle & Display */}
                        {result.heatmap && (
                            <div className="mb-8">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                                        <Lightbulb className="w-4 h-4 text-accent" />
                                        AI Focus Area
                                    </h3>
                                    <button
                                        onClick={() => setShowHeatmap(!showHeatmap)}
                                        className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all border ${showHeatmap
                                                ? 'bg-accent/20 text-accent border-accent/30'
                                                : 'bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-600'
                                            }`}
                                    >
                                        {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
                                    </button>
                                </div>

                                {showHeatmap && (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="relative rounded-xl overflow-hidden border border-white/10 aspect-square max-w-sm mx-auto"
                                    >
                                        <img
                                            src={`data:image/png;base64,${result.heatmap}`}
                                            alt="AI Heatmap Analysis"
                                            className="w-full h-full object-contain"
                                        />
                                        <div className="absolute bottom-0 left-0 right-0 p-2 bg-black/60 backdrop-blur-sm text-[10px] text-center text-slate-300">
                                            Visualizing regions contributing to prediction
                                        </div>
                                    </motion.div>
                                )}
                            </div>
                        )}

                        {/* Detailed Probabilities */}
                        <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                            <Stethoscope className="w-4 h-4 text-accent" />
                            Full Differential Diagnosis
                        </h3>
                        <div className="space-y-3">
                            {Object.entries(result.prediction)
                                .filter(([key]) => key !== result.top_finding)
                                .sort(([, a], [, b]) => b - a)
                                .slice(0, 5) // Show top 5 others
                                .map(([label, prob], idx) => (
                                    <div key={label} className="group">
                                        <div className="flex justify-between text-xs font-mono mb-1 text-slate-400">
                                            <span>{label}</span>
                                            <span>{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="w-full bg-surface h-1.5 rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${prob * 100}%` }}
                                                transition={{ duration: 1, delay: idx * 0.1 }}
                                                className={`h-full rounded-full ${prob > 0.1 ? 'bg-secondary' : 'bg-slate-700'}`}
                                            />
                                        </div>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>

                {/* Action / Disclaimer Panel */}
                <div className="space-y-6">
                    {/* Actions */}
                    <div className="bg-surface rounded-2xl p-6 border border-white/5">
                        <h3 className="text-white font-semibold mb-4">Recommended Actions</h3>
                        <div className="space-y-3">
                            <button
                                onClick={generatePDF}
                                className="w-full py-3 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg flex items-center justify-between text-sm text-slate-300 transition-colors group"
                            >
                                <span className="flex items-center gap-2">
                                    <FileText className="w-4 h-4 text-accent" />
                                    Download PDF Report
                                </span>
                                <Download className="w-4 h-4 opacity-50 group-hover:opacity-100" />
                            </button>
                            <button
                                id="consult-btn"
                                onClick={handleConsult}
                                className="w-full py-3 px-4 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-lg flex items-center justify-between text-sm text-accent transition-colors"
                            >
                                <span className="font-semibold">Consult Radiologist</span>
                                <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* HIPAA Disclaimer */}
                    <div className="bg-surface/50 rounded-xl p-4 border border-white/5 flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-slate-500 flex-shrink-0 mt-0.5" />
                        <div className="text-xs text-slate-500 leading-relaxed">
                            <strong className="text-slate-400 block mb-1">Medical Disclaimer</strong>
                            This tool uses deep learning to assist in detection but does not replace clinical diagnosis by a certified radiologist. HIPAA compliant processing.
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default ResultsPanel;
