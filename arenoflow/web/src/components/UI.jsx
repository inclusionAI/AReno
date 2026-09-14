import { t } from '../i18n';
import { ArrowUpRight, LoaderCircle, X } from 'lucide-react';
export function Button({ children, className = '', busy, ...props }) {
  return (
    <button className={`button ${className}`} {...props} disabled={busy || props.disabled}>
      {busy && <LoaderCircle size={16} className="spin" />}
      {children}
    </button>
  );
}
export function Brand({ compact = false }) {
  return (
    <a className={`brand ${compact ? 'compact' : ''}`} href="#home" aria-label={t('AReno home')}>
      <img src="/brand/logo.svg" alt="AReno" />
    </a>
  );
}
export function Badge({ status = 'draft', children }) {
  return (
    <span className={`badge ${status}`}>
      <i />
      {t(children || status.replaceAll('_', ' '))}
    </span>
  );
}
export function Empty({ icon: Icon, title, children, action }) {
  return (
    <div className="empty">
      <div className="empty-icon">{Icon && <Icon size={26} />}</div>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function PageHeader({ eyebrow, title, children, action }) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {action}
    </header>
  );
}
export function Notice({ children, error = false }) {
  return (
    <div role={error ? 'alert' : 'status'} className={`notice ${error ? 'error' : ''}`}>
      {typeof children === 'string' ? t(children) : children}
    </div>
  );
}
export function External({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="external">
      {children}
      <ArrowUpRight size={14} />
    </a>
  );
}
export function Toast({ message, onClose }) {
  return (
    message && (
      <div className="toast" role="status">
        {t(message)}
        <button aria-label={t('Dismiss')} onClick={onClose}>
          <X size={16} />
        </button>
      </div>
    )
  );
}
