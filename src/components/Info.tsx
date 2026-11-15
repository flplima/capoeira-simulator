import { CapoeiraContext } from "../context";
import { textIn } from "../data/text";
import { t, useLang } from "../i18n/lang";
import { TOPICS } from "../machine";

/** How to do the trained movements: one tab each; asking for a move opens its tab. */
export function Info() {
  const texts = CapoeiraContext.useSelector((s) => s.context.texts);
  const topic = CapoeiraContext.useSelector((s) => s.context.topic);
  const actor = CapoeiraContext.useActorRef();
  const lang = useLang();
  const source = texts[topic];
  if (!source) return null;
  const movement = textIn(source, lang);
  return (
    <aside className="info" aria-live="polite">
      <div className="tabs" role="tablist" aria-label={t("trainedMovements")}>
        {TOPICS.filter((t) => texts[t]).map((t) => (
          <button key={t} role="tab" aria-selected={t === topic} onClick={() => actor.send({ type: "SHOW", topic: t })}>
            {texts[t]!.name}
          </button>
        ))}
      </div>
      <span className="chip">{movement.category}</span>
      <h1>{movement.name}</h1>
      <p className="gloss">
        <em>{movement.translation}</em>
        {movement.aka.length > 0 && <> · {t("also")} {movement.aka.join(", ")}</>}
      </p>
      <p className="summary">{movement.summary}</p>
      <h2>{t("howTo")}</h2>
      <ol>
        {movement.steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
      {movement.tips.length > 0 && (
        <>
          <h2>{t("tips")}</h2>
          <ul>
            {movement.tips.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </>
      )}
      <h2>{t("sources")}</h2>
      <ul className="sources">
        {movement.sources.map((s) => (
          <li key={s.url}>
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>
          </li>
        ))}
      </ul>
    </aside>
  );
}
