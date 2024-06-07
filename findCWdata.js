/*
const start = async () => {
    const today = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format

    for (let i = 22000; i < 22100; i++) {
        try {
            const url = 'https://www.nytimes.com/svc/crosswords/v6/puzzle/' + i.toString() + '.json';
            const http = new XMLHttpRequest();
            http.open('GET', url, false);
            http.send(null);

            if (http.status == 200) {
                const response = JSON.parse(http.responseText);
                const publishDate = response.publicationDate;
                const subCat = response.subcategory;

                if (publishDate === today) {
                    console.log('Found matching page:', url);
                    document.getElementById("test").innerHTML = url;
                    return;
                } else {
                    console.log(url, 'publish date does not match');
                }
            } else {
                console.log(url, 'is missing');
            }
        } catch (e) {
            console.error('Error fetching URL:', url, e);
        }
    }
}

start();
*/
const extractCrosswordClues = () => {
    const clues = [];

    for (var i = 1; i < 100; i++) {
        console.log(i);
        var cell = document.getElementById("cell-id-"+i.toString());
        console.log(cell);
        var ariaLabel = cell.getAttribute('aria-label');
        if (ariaLabel) {
            clues.push(ariaLabel);
        }
    }
    return clues;
}

// Run the function to extract clues and log them to the console
const crosswordClues = extractCrosswordClues();
console.log('Crossword Clues:', crosswordClues);


