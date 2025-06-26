document
  .getElementById("registerForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Mencegah reload halaman

    let name = document.getElementById("name").value;
    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    // Ambil data pengguna dari localStorage
    let users = JSON.parse(localStorage.getItem("users")) || [];

    // Cek apakah email sudah terdaftar
    if (users.some((user) => user.email === email)) {
      alert("Email sudah digunakan!");
      return;
    }

    // Simpan pengguna baru
    users.push({ name, email, password });
    localStorage.setItem("users", JSON.stringify(users));

    alert("Pendaftaran berhasil! Silakan login.");
    window.location.href = "/login";
  });

// Fungsi menampilkan & menyembunyikan password
function togglePassword(inputId, btnId) {
  let input = document.getElementById(inputId);
  input.type = input.type === "password" ? "text" : "password";
}
