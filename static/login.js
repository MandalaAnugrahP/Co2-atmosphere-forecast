document
  .getElementById("loginForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Mencegah reload halaman

    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    // Ambil data pengguna dari localStorage
    let users = JSON.parse(localStorage.getItem("users")) || [];

    // Cek apakah email dan password cocok
    let user = users.find(
      (user) => user.email === email && user.password === password
    );

    if (user) {
      alert(`Selamat datang, ${user.name}!`);
      localStorage.setItem("loggedInUser", JSON.stringify(user));
      window.location.href = "/";
    } else {
      alert("Email atau kata sandi salah!");
    }
  });

// Fungsi menampilkan & menyembunyikan password
function togglePassword(inputId, btnId) {
  let input = document.getElementById(inputId);
  input.type = input.type === "password" ? "text" : "password";
}
