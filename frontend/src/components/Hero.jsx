import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ScanLine } from 'lucide-react';

const Hero = () => {
    return (
        <div className="relative pt-32 pb-12 px-6 overflow-hidden">
            {/* Background Elements */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-accent/5 rounded-full blur-[100px] pointer-events-none"></div>

            <div className="max-w-4xl mx-auto text-center relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-6 backdrop-blur-sm">
                        <ScanLine className="w-4 h-4 text-accent" />
                        <span className="text-xs font-mono text-accent tracking-wider uppercase">Neural Network V2.0 Active</span>
                    </div>
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.1 }}
                    className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight tracking-tight"
                >
                    Next-Gen AI <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent via-blue-400 to-secondary">
                        Disease Detection
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                    className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
                >
                    Upload chest radiographs and detect lung and thoracic diseases instantly using
                    advanced deep learning models with <span className="text-white font-semibold">99.8% accuracy detection</span>.
                </motion.p>
            </div>
        </div>
    );
};

export default Hero;
