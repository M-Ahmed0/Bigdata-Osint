document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const progressBar = document.querySelector(".progress-bar");
    const previewContainer = document.querySelector(".preview-container");
    const resultImage = document.getElementById("result-image");
    const fileInput = document.getElementById("inputfile");

    let previousFile;
    // Reset the progress bar when a new file is selected
    fileInput.addEventListener("change", function() {
      const currentFile = fileInput.files[0];
  
      if (previousFile && currentFile && previousFile.name !== currentFile.name)
        progressBar.style.width = "0%";
  
      previousFile = currentFile;
    });

    form.addEventListener("submit", function(e) {
      e.preventDefault();
  
      const file = fileInput.files[0];
  
      const formData = new FormData();
      formData.append("file", file);
  
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/");
  
      // Update the progress bar during the upload
      xhr.upload.addEventListener("progress", function(event) {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          progressBar.style.width = percentComplete + "%";
        }
      });
  
      xhr.onload = function() {
        if (xhr.status === 200) {
          // Handle the successful response
          const response = JSON.parse(xhr.responseText);
          resultImage.src = response.image_path;
  
          // Display the preview image
          const previewImage = document.createElement("img");
          previewImage.src = response.image_path;
          previewImage.className = "preview-image";
          previewContainer.appendChild(previewImage);
        } else {
          // Handle the error response
          console.error(xhr.statusText);
        }
  
        // Reset the form and progress bar
        form.reset();
        progressBar.style.width = "0%";
        fileInput.value = "";
      };
  
      // Send the form data to the server
      xhr.send(formData);
    });
  });
  