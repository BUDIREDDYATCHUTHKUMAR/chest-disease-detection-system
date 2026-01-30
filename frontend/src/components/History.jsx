import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, Calendar, FileText, Activity, Trash2, ChevronRight } from 'lucide-react';
import { format } from 'date-fns';

const History = ({ history, onSelect, onClear }) => {
    if (history.length === 0) {
        return (
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col items-center justify-center min-h-[50vh] text-slate-500"
            >
                <Clock className="w-12 h-12 mb-4 opacity-50" />
                <h3 className="text-lg font-medium text-slate-400">No History Available</h3>
                <p className="text-sm">Past analysis results will appear here.</p>
            </motion.div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto mt-8">
            <div className="flex items-center justify-between mb-8">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <HistoryIcon className="w-5 h-5 text-accent" />
                    Scan History
                </h2>
                <button
                    onClick={onClear}
                    className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1 px-3 py-1 rounded-full hover:bg-red-500/10 transition-colors"
                >
                    <Trash2 className="w-3 h-3" />
                    Clear Log
                </button>
            </div>

            <div className="space-y-4">
                {history.map((item, index) => (
                    <motion.div
                        key={item.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="group glass-panel p-4 rounded-xl flex items-center justify-between hover:border-accent/30 transition-all cursor-pointer"
                        onClick={() => onSelect(item)}
                    >
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 bg-black/50 rounded-lg overflow-hidden border border-slate-700">
                                <img src={item.preview} alt="X-ray" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                            </div>
                            <div>
                                <h4 className="text-white font-mono font-bold text-lg">{item.result.top_finding}</h4>
                                <div className="flex items-center gap-4 text-xs text-slate-400 mt-1">
                                    <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {format(new Date(item.timestamp), 'MMM dd, yyyy')}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        {format(new Date(item.timestamp), 'HH:mm')}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center gap-6">
                            <div className="text-right">
                                <div className="text-xl font-mono font-bold text-accent">
                                    {(item.result.top_probability * 100).toFixed(1)}%
                                </div>
                                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Confidence</div>
                            </div>
                            <div className="w-8 h-8 rounded-full bg-surface border border-slate-700 flex items-center justify-center group-hover:bg-accent group-hover:text-primary transition-colors">
                                <ChevronRight className="w-4 h-4" />
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

// Simple wrapper icon to avoid conflict with standard library name if needed
const HistoryIcon = (props) => (
    <Activity {...props} />
);

export default History;
