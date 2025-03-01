document.getElementById("check-btn").addEventListener("click", function() {
    const newsText = document.getElementById("news-input").value;
    fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "text": newsText })
    })
    .then(response => response.json())
    .then(data => {
        const result = data.prediction === 1 ? "Fake News" : "Real News";
        document.getElementById("result").innerText = `Prediction: ${result}`;
    })
    .catch(error => console.error('Error:', error));
});
