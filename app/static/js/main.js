const imageInput = document.querySelector('#leaf-image');
const fileName = document.querySelector('#file-name');
const video = document.querySelector('#camera-preview');
const canvas = document.querySelector('#camera-capture');
const cameraStatus = document.querySelector('#camera-status');
const startCamera = document.querySelector('#start-camera');
const capturePhoto = document.querySelector('#capture-photo');
const retakePhoto = document.querySelector('#retake-photo');
let cameraStream = null;

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  if (video) video.srcObject = null;
}

if (imageInput && fileName) {
  imageInput.addEventListener('change', () => {
    fileName.textContent = imageInput.files?.[0]?.name ?? 'No image selected';
    if (imageInput.files?.[0] && canvas) {
      stopCamera();
      const image = new Image();
      image.onload = () => {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.getContext('2d').drawImage(image, 0, 0);
        canvas.hidden = false;
        if (video) video.hidden = true;
        if (startCamera) startCamera.hidden = true;
        if (capturePhoto) capturePhoto.hidden = true;
        if (retakePhoto) retakePhoto.hidden = false;
      };
      image.src = URL.createObjectURL(imageInput.files[0]);
    }
  });
}

if (startCamera && video && capturePhoto && canvas && imageInput) {
  startCamera.addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      if (cameraStatus) cameraStatus.textContent = 'Camera access needs a secure browser context. Use localhost or choose a photo instead.';
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      video.srcObject = cameraStream;
      video.hidden = false;
      canvas.hidden = true;
      await video.play();
      startCamera.hidden = true;
      capturePhoto.hidden = false;
      if (retakePhoto) retakePhoto.hidden = true;
      if (cameraStatus) cameraStatus.textContent = 'Frame the leaf and capture when it is in focus.';
    } catch {
      if (cameraStatus) cameraStatus.textContent = 'Camera permission was unavailable. Choose a photo or check your browser permissions.';
    }
  });

  capturePhoto.addEventListener('click', async () => {
    if (!video.videoWidth || !video.videoHeight) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    if (!imageBlob) return;
    const transfer = new DataTransfer();
    transfer.items.add(new File([imageBlob], `fieldnote-capture-${Date.now()}.jpg`, { type: 'image/jpeg' }));
    imageInput.files = transfer.files;
    fileName.textContent = imageInput.files[0].name;
    stopCamera();
    video.hidden = true;
    canvas.hidden = false;
    capturePhoto.hidden = true;
    if (retakePhoto) retakePhoto.hidden = false;
    if (cameraStatus) cameraStatus.textContent = 'Preview captured. Retake or analyze this image.';
  });

  retakePhoto?.addEventListener('click', () => {
    stopCamera();
    imageInput.value = '';
    fileName.textContent = 'No image selected';
    canvas.hidden = true;
    video.hidden = true;
    retakePhoto.hidden = true;
    startCamera.hidden = false;
    if (capturePhoto) capturePhoto.hidden = true;
    if (cameraStatus) cameraStatus.textContent = 'Use a camera or choose a photo.';
  });
}

document.querySelectorAll('[data-clear-offline]').forEach((form) => {
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    localStorage.removeItem('fieldnote-reminder-queue');
    const request = indexedDB.deleteDatabase('fieldnote-offline');
    const submitForm = () => form.submit();
    request.onsuccess = submitForm;
    request.onerror = submitForm;
    request.onblocked = submitForm;
  });
});

if ('serviceWorker' in navigator && window.isSecureContext) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/service-worker.js').catch(() => {}));
}