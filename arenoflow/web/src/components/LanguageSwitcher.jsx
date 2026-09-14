import { setLanguage, t, useLanguage } from '../i18n';

export default function LanguageSwitcher() {
  const language = useLanguage();
  return (
    <div className="language-switcher" role="group" aria-label={t('Language')}>
      <button
        type="button"
        lang="en"
        aria-pressed={language === 'en'}
        onClick={() => setLanguage('en')}
      >
        EN
      </button>
      <button
        type="button"
        lang="zh-CN"
        aria-pressed={language === 'zh'}
        onClick={() => setLanguage('zh')}
      >
        中文
      </button>
    </div>
  );
}
