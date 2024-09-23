// Menampilkan input latar belakang hanya jika opsi 'Ganti Latar Belakang' dipilih
function toggleBackgroundInput() {
    const processType = document.getElementById('process_type').value;
    const backgroundInputContainer = document.getElementById('backgroundInputContainer');
    if (processType === 'change_background') {
        backgroundInputContainer.style.display = 'block';
    } else {
        backgroundInputContainer.style.display = 'none';
    }
}

// Saat halaman selesai dimuat, jalankan fungsi untuk memeriksa opsi yang dipilih
window.onload = function() {
    toggleBackgroundInput();  // Periksa opsi yang dipilih saat halaman dimuat
};

// Menampilkan pesan "Mohon Tunggu" saat form disubmit
function showLoading() {
    const loadingMessage = document.getElementById('loadingMessage');
    loadingMessage.style.display = 'block';  // Tampilkan pesan loading
}
