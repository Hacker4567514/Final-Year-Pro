function performSearch() {
    const location = document.getElementById('location').value;
    const sector = document.getElementById('sector').value;
    const stage = document.getElementById('stage').value;
    const category = document.getElementById('category').value;
    const rating = document.getElementById('rating').value;
    const reviews = document.getElementById('reviews').value;

    // Send data to backend API
    fetch('/predict', {  // Ensure this matches your Flask route
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ location, sector, stage, category, rating, reviews })
    })
    .then(response => response.json())
    .then(data => {
        // Display result in the HTML element with ID "result"
        document.getElementById('result').innerText = `Prediction: ${data.result}`;
    })
    .catch(error => console.error('Error:', error));
}
