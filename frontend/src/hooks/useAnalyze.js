import { useState, useEffect, useRef } from 'react';
import { analyzeMessage } from '../services/api';

/**
 * Custom hook for message analysis with debouncing
 * 
 * @param {string} message - The message to analyze
 * @param {number} debounceMs - Debounce delay in milliseconds
 * @returns {Object} { analysis, isAnalyzing }
 */
export function useAnalyze(message, debounceMs = 500) {
    const [analysis, setAnalysis] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const timeoutRef = useRef(null);
    const abortControllerRef = useRef(null);

    useEffect(() => {
        // Clear previous timeout
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }

        // Abort previous request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Don't analyze empty messages
        if (!message || message.trim().length === 0) {
            setAnalysis(null);
            setIsAnalyzing(false);
            return;
        }

        // Start loading indicator immediately
        setIsAnalyzing(true);

        // Debounce the API call
        timeoutRef.current = setTimeout(async () => {
            try {
                abortControllerRef.current = new AbortController();
                const result = await analyzeMessage(message);
                setAnalysis(result);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Analysis error:', error);
                }
            } finally {
                setIsAnalyzing(false);
            }
        }, debounceMs);

        // Cleanup
        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, [message, debounceMs]);

    return { analysis, isAnalyzing };
}
