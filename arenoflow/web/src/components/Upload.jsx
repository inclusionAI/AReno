import { t } from '../i18n';
import { useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { api } from '../api';
import { Notice } from './UI';
export default function Upload({ onUploaded, onBusyChange }) {
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState(''),
    [error, setError] = useState(false),
    [busy, setBusy] = useState(false);
  async function upload(file) {
    if (!file) return;
    if (!/\.(jsonl?|csv|tsv|parquet|arrow)$/i.test(file.name)) {
      setError(true);
      setMessage(
        t(
          'Choose a JSON, JSONL, CSV, TSV, Parquet or Arrow dataset. Upload media in Media attachments.',
        ),
      );
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      setError(true);
      setMessage(t('Upload up to 16 MiB, or use a remote dataset repository for larger data.'));
      return;
    }
    setBusy(true);
    onBusyChange?.(true);
    setError(false);
    setMessage(t('Staging dataset locally…'));
    try {
      const result = await uploadFile(file);
      onUploaded(result.path, result);
      setMessage(
        t('{p0} staged. It will be uploaded to your Modal Volume when you launch.', {
          p0: result.name,
        }),
      );
    } catch (e) {
      setError(true);
      setMessage(e.message || 'File could not be read');
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }
  return (
    <div
      className={`upload-dataset dataset-dropzone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (busy) return;
        const files = [...e.dataTransfer.files];
        if (files.length !== 1) {
          setError(true);
          setMessage(t('Choose one dataset file. Add media in Media attachments below.'));
          return;
        }
        upload(files[0]);
      }}
    >
      <UploadCloud size={28} className="dropzone-icon" />
      <strong>{t('Drop your dataset here')}</strong>
      <p>
        {t('Choose a data file from your computer. Add image, audio and video attachments below.')}
      </p>
      <label className="button">
        <UploadCloud size={16} />
        {busy ? t('Uploading…') : t('Browse local files')}
        <input
          type="file"
          aria-label={t('Upload dataset')}
          accept=".json,.jsonl,.csv,.tsv,.parquet,.arrow"
          disabled={busy}
          onChange={(e) => {
            upload(e.target.files[0]);
            e.target.value = '';
          }}
        />
      </label>
      <small>{t('JSONL, JSON, CSV, TSV, Parquet or Arrow · up to 16 MiB')}</small>
      {message && <Notice error={error}>{message}</Notice>}
    </div>
  );
}
export async function uploadFile(file) {
  if (file.size > 16 * 1024 * 1024)
    throw new Error(
      'Each uploaded file can be up to 16 MiB. Use a dataset repository for larger media.',
    );
  const content = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = () => reject(new Error('Could not read file'));
    reader.readAsDataURL(file);
  });
  return api('/uploads', {
    name: file.name,
    content,
  });
}
