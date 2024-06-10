const extractCurrentClue = () => {

    // Select the currently selected cell
    const selectedCell = document.querySelector('div.xwd__board--content g.xwd__cell rect.xwd__cell--selected');

    if (!selectedCell) {
        console.error('No selected cell found');
        return null;
    }
    
    const ariaLabel = selectedCell.getAttribute('aria-label');

    // Example aria-label format: "58D: Short-tailed weasel, Answer: 5 letters, Letter: 0"
    const clueRegex = /(\d+[A-Z]): (.*), Answer: (\d+) letters/;
    const match = ariaLabel.match(clueRegex);

    if (match) {
        const clueNumber = match[1];
        const clueText = match[2];
        const answerLength = match[3];

        return {
            clue: clueText,
            length: parseInt(answerLength)
        };
    } else {
        console.error('Clue and answer length not found in aria-label');
        return null;
    }
};


// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'extractCurrentClue') {
        const currentClue = extractCurrentClue();
        sendResponse(currentClue);
    } else if (request.action === 'log') {
        console.log("FROM POPUP:", request.message);
    } else if (request.action === 'error') {
        console.error("FROM POPUP:", request.message);
    }
});
