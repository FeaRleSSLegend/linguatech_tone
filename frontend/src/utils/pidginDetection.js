/**
 * 🔥 PIDGIN DETECTION UTILITY - Presentation "Wow" Factor
 * 
 * Common Nigerian Pidgin insults and aggressive phrases for demo purposes
 * This demonstrates the "Resonance Shield" capability
 */

export const PIDGIN_FLAGGED_WORDS = [
    // Insults
    'mumu',
    'olodo',
    'ode',
    'ewu',
    'werey',
    'agidi',
    'oniranu',
    'olofofo',
    'ashawo',
    'useless person',
    'foolish person',
    
    // Aggressive phrases
    'na you sabi',
    'comot',
    'getat',
    'shege',
    'yeye',
    'nonsense',
    'rubbish',
    'idiot',
    'stupid',
    'mad person',
    'crazy person',
    
    // Threats
    'I go beat you',
    'make I slap you',
    'thunder fire you',
    'you be goat',
    'you dey craze',
    
    // Cultural specific
    'abeg comot',
    'wetin you dey talk',
    'sharp sharp',
    'no dull'
];

/**
 * Check if text contains Pidgin flagged words
 * @param {string} text - Message text to check
 * @returns {boolean} - True if flagged content detected
 */
export function containsPidginFlags(text) {
    if (!text || typeof text !== 'string') return false;
    
    const lowerText = text.toLowerCase();
    
    return PIDGIN_FLAGGED_WORDS.some(word => {
        // Check for whole word matches to avoid false positives
        const regex = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
        return regex.test(lowerText);
    });
}

/**
 * Get the specific flagged words found in text
 * @param {string} text - Message text to check
 * @returns {string[]} - Array of flagged words found
 */
export function getFlaggedWords(text) {
    if (!text || typeof text !== 'string') return [];
    
    const lowerText = text.toLowerCase();
    const found = [];
    
    PIDGIN_FLAGGED_WORDS.forEach(word => {
        const regex = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
        if (regex.test(lowerText)) {
            found.push(word);
        }
    });
    
    return found;
}

/**
 * Get moderation level based on flagged content
 * @param {string} text - Message text to check
 * @returns {Object} - Moderation info
 */
export function getModerationInfo(text) {
    const flaggedWords = getFlaggedWords(text);
    
    if (flaggedWords.length === 0) {
        return {
            isFlagged: false,
            level: 'clean',
            flaggedWords: [],
            message: null
        };
    }
    
    return {
        isFlagged: true,
        level: flaggedWords.length >= 3 ? 'severe' : 'moderate',
        flaggedWords,
        message: `Detected Pidgin/aggressive language: ${flaggedWords.join(', ')}`
    };
}