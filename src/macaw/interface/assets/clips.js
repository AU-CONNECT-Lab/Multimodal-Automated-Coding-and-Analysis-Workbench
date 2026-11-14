// A voice's clip: the round button plays or pauses the audio beside it, one clip at a time.
(() => {
  document.addEventListener('click', (e) => {
    const button = e.target.closest('.clip-play');
    if (!button) return;
    const audio = button.parentElement.querySelector('audio');
    if (!audio) return;
    document.querySelectorAll('.clip audio').forEach((other) => { if (other !== audio && !other.paused) other.pause(); });
    if (audio.paused) audio.play(); else audio.pause();
  });
  const mark = (e) => {
    if (!(e.target instanceof HTMLAudioElement)) return;
    const button = e.target.parentElement && e.target.parentElement.querySelector('.clip-play');
    if (button) button.classList.toggle('on', !e.target.paused && !e.target.ended);
  };
  ['play', 'pause', 'ended'].forEach((type) => document.addEventListener(type, mark, true));
})();
