document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("signup-form");
    const togglePassword = document.getElementById("toggle-password");
    const passwordInput = document.getElementById("password");
  
    // Toggle password visibility
    togglePassword.addEventListener("click", function () {
      if (passwordInput.type === "password") {
        passwordInput.type = "text";
        togglePassword.innerHTML = "&#128064;"; // Ubah ikon mata
      } else {
        passwordInput.type = "password";
        togglePassword.innerHTML = "&#128065;"; // Ubah kembali ke ikon default
      }
    });
  
    // Handle form submission
    form.addEventListener("submit", function (event) {
      event.preventDefault();
  
      const name = document.getElementById("name").value;
      const mobile = document.getElementById("mobile").value;
      const email = document.getElementById("email").value;
      const password = document.getElementById("password").value;
  
      if (name && mobile && email && password) {
        // Simpan data ke localStorage
        const userData = {
          name: name,
          mobile: mobile,
          email: email,
          password: password, // Sebaiknya dienkripsi dalam aplikasi nyata
        };
  
        localStorage.setItem("user", JSON.stringify(userData));
        alert("Sign Up Successful! Data saved in localStorage.");
        window.location.href = "Login.html";
      } else {
        alert("Please fill in all fields!");
      }
    });
  });
  