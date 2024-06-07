const sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}

const start = async () => {
    const today = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format

    for (let i = 1; i < 19423; i++) {
        await sleep(2000);
        try {
            const url = 'https://www.nytimes.com/svc/crosswords/v6/puzzle/' + i.toString() + '.json';
            const http = new XMLHttpRequest();
            http.open('GET', url, false);
            http.send(null);

            if (http.status == 200) {
                const response = JSON.parse(http.responseText);
                const publishDate = response.publish_date; // Adjust this based on the actual key in the JSON response

                if (publishDate === today) {
                    console.log('Found matching page:', url);
                    window.location.href = url;
                    return; // Stop the loop as we found the matching page
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
