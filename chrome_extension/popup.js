const log = (message) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'log', message: message });
    });
};

const error = (message) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'error', message: message });
    });
};

document.addEventListener('DOMContentLoaded', () => {
    const getClueBtn = document.getElementById('get-clue-btn');
    const clueElement = document.getElementById('clue');
    const lengthElement = document.getElementById('length');

    if (getClueBtn) {
        getClueBtn.addEventListener('click', () => {

            // Send a message to the content script to extract the current clue
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                chrome.tabs.sendMessage(tabs[0].id, { action: 'extractCurrentClue' }, (response) => {
                    if (response) {
                        log(`Clue: ${response.clue}, Length: ${response.length}`);
                        clueElement.textContent = response.clue;
                        lengthElement.textContent = response.length;
                    } else {
                        error('No response from content script.');
                        clueElement.textContent = "Error retrieving clue.";
                        lengthElement.textContent = "";
                    }
                });
            });
        });
    } else {
        console.error("Button not found in popup.");
    }
});
