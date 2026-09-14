import { t } from '../i18n';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags } from '@lezer/highlight';
const colors = HighlightStyle.define([
  {
    tag: tags.keyword,
    color: '#7442a3',
    fontWeight: '600',
  },
  {
    tag: [tags.function(tags.variableName), tags.definition(tags.variableName)],
    color: '#2455bb',
  },
  {
    tag: [tags.string, tags.special(tags.string)],
    color: '#167253',
  },
  {
    tag: [tags.number, tags.bool, tags.null],
    color: '#996020',
  },
  {
    tag: tags.comment,
    color: '#68778d',
    fontStyle: 'italic',
  },
  {
    tag: [tags.operator, tags.punctuation],
    color: '#58677f',
  },
  {
    tag: tags.typeName,
    color: '#176881',
  },
]);
const theme = EditorView.theme({
  '&': {
    backgroundColor: 'var(--paper)',
    color: 'var(--ink)',
    fontSize: '13px',
  },
  '.cm-content': {
    padding: '16px 0',
    caretColor: 'var(--brand)',
  },
  '.cm-scroller': {
    fontFamily: "'SFMono-Regular', Consolas, monospace",
    lineHeight: '1.8',
  },
  '.cm-gutters': {
    backgroundColor: 'var(--inset)',
    color: 'var(--muted)',
    borderRight: '1px solid var(--line)',
  },
  '.cm-activeLine, .cm-activeLineGutter': {
    backgroundColor: 'var(--brand-tint)',
  },
  '&.cm-focused': {
    outline: 'none',
  },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
    backgroundColor: '#dce7fc',
  },
  '.cm-cursor': {
    borderLeftColor: 'var(--brand)',
  },
});
const extensions = [
  python(),
  syntaxHighlighting(colors),
  EditorView.lineWrapping,
  EditorView.contentAttributes.of({
    'aria-label': 'Python code',
    spellcheck: 'false',
  }),
];
export default function PythonEditor({ value, onChange }) {
  return (
    <div className="python-code-editor">
      <div className="code-editor-heading">
        <span>{t('Python code')}</span>
        <small>{t('Python · 4-space indentation')}</small>
      </div>
      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={extensions}
        theme={theme}
        minHeight="360px"
        maxHeight="600px"
        indentWithTab
        basicSetup={{
          tabSize: 4,
        }}
      />
      <small className="code-editor-help">
        {t('Tab to indent · Escape, then Tab to move focus out of the editor')}
      </small>
    </div>
  );
}
