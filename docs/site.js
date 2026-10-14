const root = document.documentElement;
const themeButton = document.getElementById('theme-toggle');
const savedTheme = localStorage.getItem('mulag-theme');
if (savedTheme === 'dark' || (!savedTheme && matchMedia('(prefers-color-scheme: dark)').matches)) root.dataset.theme = 'dark';
function syncThemeButton() {
  const dark = root.dataset.theme === 'dark';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} appearance`);
  themeButton.querySelector('.theme-label').textContent = dark ? 'Light' : 'Dark';
}
syncThemeButton();
themeButton.addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('mulag-theme', root.dataset.theme);
  syncThemeButton();
});

const video = document.getElementById('research-video');
const caption = document.getElementById('video-caption');
const views = {
  front: { src: 'https://raw.githubusercontent.com/kevm98/mulag_development/main/videos/adaptive_rollout_mesh1337_front_view.mp4', caption: 'Front view · S-curve trajectory on mesh 1337. Recorded as a separate run from the back view.' },
  back: { src: 'https://raw.githubusercontent.com/kevm98/mulag_development/main/videos/adaptive_rollout_mesh1337_back_view.mp4', caption: 'Back view · Meander trajectory on mesh 1337. Recorded as a separate run from the front view.' }
};
document.querySelectorAll('[data-video]').forEach(button => button.addEventListener('click', () => {
  const selected = views[button.dataset.video];
  if (!selected || video.currentSrc === selected.src) return;
  video.pause();
  video.querySelector('source').src = selected.src;
  video.load();
  video.setAttribute('aria-label', `Simulated MULAG verge mower ${button.dataset.video}-view rollout`);
  caption.textContent = selected.caption;
  document.querySelectorAll('[data-video]').forEach(item => {
    const active = item === button;
    item.classList.toggle('is-active', active);
    item.setAttribute('aria-pressed', String(active));
  });
}));

document.getElementById('copy-citation').addEventListener('click', async event => {
  const content = document.querySelector('#bibtex code').textContent;
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(content);
    event.currentTarget.textContent = 'Copied';
    status.textContent = 'BibTeX copied to clipboard';
    setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Command-C or Control-C to copy it.';
  }
});
