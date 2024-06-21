document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const input = document.getElementById('imageInput');
    if (input.files.length === 0) {
        alert('Por favor, selecione uma imagem.');
        return;
    }

    const formData = new FormData();
    formData.append('file', input.files[0]);

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert('Upload feito com sucesso!');
        console.log(data);
    })
    .catch(error => {
        console.error('Erro:', error);
        alert('Erro ao fazer upload da imagem.');
    });
});
