import { LANGS, setLang, t, useLang, type Lang } from "../i18n/lang";

const NAMES: Record<Lang, string> = { en: "English", pt: "Português" };

/** The United States flag, simplified to four stripes and a canton. */
const UnitedStates = () => (
  <svg viewBox="0 0 20 14" width="20" height="14" aria-hidden>
    <rect width="20" height="14" fill="#fff" />
    {[0, 4, 8, 12].map((y) => (
      <rect key={y} y={y} width="20" height="2" fill="#b22234" />
    ))}
    <rect width="9" height="8" fill="#3c3b6e" />
    {[1.8, 4.5, 7.2].flatMap((x) => [1.6, 4, 6.4].map((y) => <circle key={`${x}-${y}`} cx={x} cy={y} r="0.55" fill="#fff" />))}
  </svg>
);

/** The Brazilian flag: green field, yellow diamond, blue disc with its white band. */
const Brazil = () => (
  <svg viewBox="0 0 20 14" width="20" height="14" aria-hidden>
    <rect width="20" height="14" fill="#009c3b" />
    <path d="M10 1.4 18.4 7 10 12.6 1.6 7Z" fill="#ffdf00" />
    <circle cx="10" cy="7" r="3.3" fill="#002776" />
    <path d="M6.8 6.2c2.2-.7 4.6-.2 6.4 1.2" stroke="#fff" strokeWidth="0.7" fill="none" />
  </svg>
);

const FLAGS: Record<Lang, () => React.JSX.Element> = { en: UnitedStates, pt: Brazil };

/** Two flag buttons in a segmented control: English and Portuguese. */
export function LanguageSwitch() {
  const lang = useLang();
  return (
    <div className="segmented lang-switch" role="group" aria-label={t("languageGroup")}>
      {LANGS.map((l) => {
        const Flag = FLAGS[l];
        return (
          <button key={l} aria-pressed={l === lang} aria-label={NAMES[l]} title={NAMES[l]} onClick={() => setLang(l)}>
            <Flag />
          </button>
        );
      })}
    </div>
  );
}
