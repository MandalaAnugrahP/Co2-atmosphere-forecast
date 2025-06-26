document.addEventListener("DOMContentLoaded", function () {
    const username = localStorage.getItem("username");
    const email = localStorage.getItem("email");

    if (!username || !email) {
        alert("Anda Sudah Keluar");
        window.location.href = "/login";
        return;
    }

    document.getElementById("username").textContent = username;
    document.getElementById("email").textContent = email;

    // Fungsi untuk logout
    document.getElementById("logout-button").addEventListener("click", function () {
        localStorage.removeItem("username");
        localStorage.removeItem("email");
        alert("Anda telah keluar.");
        window.location.href = "/login";
    });
});
