import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './styles.css'

const API_BASE = import.meta.env.VITE_API_BASE
const RAPIDAPI_KEY = import.meta.env.VITE_RAPIDAPI_KEY

function ScoreRing({ score, label, color }) {
  const r = 36
  const circ = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, score || 0))
  const dash = (pct / 100) * circ
  return (
    <div className="score-ring-wrap">
      <svg width="96" height="96" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r={r} fill="none" stroke="#1e1e1e" strokeWidth="7" />
        <circle
          cx="48" cy="48" r={r} fill="none"
          stroke={color} strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 48 48)"
          className="ring-fill"
        />
        <text x="48" y="48" dominantBaseline="central" textAnchor="middle"
          fill={color} fontSize="15" fontWeight="700" fontFamily="'DM Mono', monospace">
          {Math.round(pct)}
        </text>
      </svg>
      <span className="ring-label">{label}</span>
    </div>
  )
}

function TagList({ items, color }) {
  if (!items?.length) return <span className="muted-text">None identified</span>
  return (
    <div className="tag-list">
      {items.map((t, i) => (
        <span key={i} className="tag" style={{ '--tag-color': color }}>{t}</span>
      ))}
    </div>
  )
}

function AnalysisCard({ title, icon, children, accent }) {
  const ref = useRef()
  useEffect(() => {
    if (!ref.current) return
    if (typeof gsap === 'undefined') return
    gsap.fromTo(ref.current,
      { opacity: 0, y: 32 },
      { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out',
        scrollTrigger: { trigger: ref.current, start: 'top 90%', once: true } }
    )
  }, [])
  return (
    <div className="analysis-card" ref={ref} style={{ '--card-accent': accent }}>
      <div className="card-header">
        <span className="card-icon">{icon}</span>
        <h2 className="card-title">{title}</h2>
        <div className="card-line" />
      </div>
      {children}
    </div>
  )
}

function RecommendList({ items }) {
  if (!items?.length) return null
  return (
    <div className="recom-list">
      {items.map((r, i) => (
        <div key={i} className="recom-item">
          <span className="recom-num">{String(i + 1).padStart(2, '0')}</span>
          <span className="recom-text">{r}</span>
        </div>
      ))}
    </div>
  )
}

function ParsedResumePanel({ data }) {
  const [open, setOpen] = useState(false)
  const ref = useRef()
  useEffect(() => {
    if (!ref.current || typeof gsap === 'undefined') return
    gsap.fromTo(ref.current,
      { opacity: 0, y: 24 },
      { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out', delay: 0.1 }
    )
  }, [])
  if (!data) return null
  return (
    <div className="parsed-panel" ref={ref}>
      <button className="parsed-toggle" onClick={() => setOpen(o => !o)}>
        <span>Parsed Resume Data</span>
        <span className="toggle-arrow">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="parsed-body">
          <div className="parsed-grid">
            <div className="parsed-field"><span className="pf-label">Name</span><span className="pf-val">{data.full_name || '—'}</span></div>
            <div className="parsed-field"><span className="pf-label">Email</span><span className="pf-val">{data.email || '—'}</span></div>
            <div className="parsed-field"><span className="pf-label">Phone</span><span className="pf-val">{data.phone || '—'}</span></div>
            <div className="parsed-field"><span className="pf-label">Location</span><span className="pf-val">{data.location || '—'}</span></div>
          </div>
          {data.summary && <p className="parsed-summary">{data.summary}</p>}
          {data.skills?.length > 0 && (
            <div className="parsed-section">
              <span className="ps-label">Skills</span>
              <TagList items={data.skills} color="var(--amber)" />
            </div>
          )}
          {data.achievements?.length > 0 && (
            <div className="parsed-section">
              <span className="ps-label">Achievements</span>
              {data.achievements.map((a, i) => (
                <div key={i} className="achievement-item">
                  <strong>{a.title}</strong>
                  {a.description && <span> — {a.description}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Job Card ──────────────────────────────────────────────── */
function JobCard({ job, index }) {
  const ref = useRef()
  useEffect(() => {
    if (!ref.current || typeof gsap === 'undefined') return
    gsap.fromTo(ref.current,
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.45, ease: 'power3.out', delay: index * 0.07 }
    )
  }, [])

  const posted = job.job_posted_at_datetime_utc
    ? new Date(job.job_posted_at_datetime_utc).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
    : null

  const salaryParts = []
  if (job.job_min_salary) salaryParts.push(`₹${Number(job.job_min_salary).toLocaleString('en-IN')}`)
  if (job.job_max_salary) salaryParts.push(`₹${Number(job.job_max_salary).toLocaleString('en-IN')}`)
  const salary = salaryParts.length ? salaryParts.join(' – ') + (job.job_salary_period ? ` / ${job.job_salary_period}` : '') : null

  const typeColor = {
    FULLTIME: '#34d399',
    PARTTIME: '#60a5fa',
    CONTRACTOR: '#f472b6',
    INTERN: '#f59e0b',
  }[job.job_employment_type] || '#888'

  return (
    <div className="job-card" ref={ref}>
      <div className="job-card-top">
        {job.employer_logo
          ? <img src={job.employer_logo} alt={job.employer_name} className="job-logo" onError={e => { e.target.style.display = 'none' }} />
          : <div className="job-logo-fallback">{(job.employer_name || '?')[0]}</div>
        }
        <div className="job-meta">
          <span className="job-company">{job.employer_name || 'Company'}</span>
          <span className="job-location">
            {[job.job_city, job.job_state, job.job_country].filter(Boolean).join(', ') || 'Location N/A'}
            {job.job_is_remote && <span className="job-remote-badge">Remote</span>}
          </span>
        </div>
        {job.job_employment_type && (
          <span className="job-type-badge" style={{ '--type-color': typeColor }}>
            {job.job_employment_type.replace('FULLTIME', 'Full-time').replace('PARTTIME', 'Part-time').replace('CONTRACTOR', 'Contract').replace('INTERN', 'Intern')}
          </span>
        )}
      </div>

      <h3 className="job-title">{job.job_title}</h3>

      {job.job_description && (
        <p className="job-desc">{job.job_description.slice(0, 200).trim()}{job.job_description.length > 200 ? '…' : ''}</p>
      )}

      <div className="job-card-footer">
        <div className="job-footer-left">
          {salary && <span className="job-salary">{salary}</span>}
          {posted && <span className="job-posted">Posted {posted}</span>}
        </div>
        {job.job_apply_link && (
          <a href={job.job_apply_link} target="_blank" rel="noopener noreferrer" className="job-apply-btn">
            Apply →
          </a>
        )}
      </div>
    </div>
  )
}

/* ── Jobs Section ──────────────────────────────────────────── */
function JobsSection({ query, inferredRole }) {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [page, setPage] = useState(1)
  const [filter, setFilter] = useState('ALL')
  const JOBS_PER_PAGE = 6

  useEffect(() => {
    if (!query) return
    setLoading(true)
    setError('')
    const options = {
      method: 'GET',
      url: 'https://jsearch.p.rapidapi.com/search',
      params: {
        query,
        page: '1',
        num_pages: '50',
        country: 'in',
        date_posted: 'all',
        location: 'india',
      },
      headers: {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': 'jsearch.p.rapidapi.com',
      }
    }
    axios.request(options)
      .then(res => {
        setJobs(res.data?.data || [])
      })
      .catch(err => {
        setError(err.message || 'Failed to fetch jobs')
      })
      .finally(() => setLoading(false))
  }, [query])

  const employmentTypes = ['ALL', ...Array.from(new Set(jobs.map(j => j.job_employment_type).filter(Boolean)))]
  const filtered = filter === 'ALL' ? jobs : jobs.filter(j => j.job_employment_type === filter)
  const totalPages = Math.ceil(filtered.length / JOBS_PER_PAGE)
  const paginated = filtered.slice((page - 1) * JOBS_PER_PAGE, page * JOBS_PER_PAGE)

  return (
    <section className="jobs-section" id="jobs">
      <div className="jobs-section-header">
        <div className="jobs-title-row">
          <span className="jobs-icon">◈</span>
          <h2 className="jobs-heading">Job Matches</h2>
          <div className="jobs-role-pill">{inferredRole || query}</div>
          <div className="card-line" style={{ flex: 1 }} />
        </div>
        <p className="jobs-sub">Live job listings in India matching your resume profile</p>

        {employmentTypes.length > 1 && (
          <div className="jobs-filter-row">
            {employmentTypes.map(t => (
              <button
                key={t}
                className={`jobs-filter-btn ${filter === t ? 'active' : ''}`}
                onClick={() => { setFilter(t); setPage(1) }}
              >
                {t === 'ALL' ? 'All' : t.replace('FULLTIME', 'Full-time').replace('PARTTIME', 'Part-time').replace('CONTRACTOR', 'Contract').replace('INTERN', 'Intern')}
              </button>
            ))}
          </div>
        )}
      </div>

      {loading && (
        <div className="jobs-loading">
          <span className="spinner" style={{ borderTopColor: '#f59e0b', borderColor: 'rgba(245,158,11,0.2)' }} />
          <span>Fetching live job listings…</span>
        </div>
      )}

      {error && !loading && (
        <div className="error-bar" style={{ maxWidth: '100%' }}>Failed to load jobs: {error}</div>
      )}

      {!loading && !error && paginated.length === 0 && (
        <div className="jobs-empty">No job listings found for this role.</div>
      )}

      {!loading && !error && paginated.length > 0 && (
        <>
          <div className="jobs-count">{filtered.length} listing{filtered.length !== 1 ? 's' : ''} found</div>
          <div className="jobs-grid">
            {paginated.map((job, i) => (
              <JobCard key={job.job_id || i} job={job} index={i} />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="jobs-pagination">
              <button
                className="jobs-page-btn"
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >← Prev</button>
              <span className="jobs-page-info">{page} / {totalPages}</span>
              <button
                className="jobs-page-btn"
                disabled={page === totalPages}
                onClick={() => setPage(p => p + 1)}
              >Next →</button>
            </div>
          )}
        </>
      )}
    </section>
  )
}

/* ── Main App ──────────────────────────────────────────────── */
export default function App() {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const heroRef = useRef()
  const formRef = useRef()
  const inputRef = useRef()

  useEffect(() => {
    if (typeof gsap === 'undefined') return
    const tl = gsap.timeline()
    tl.fromTo('.hero-eyebrow', { opacity: 0, y: -16 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out' })
      .fromTo('.hero-title span', { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', stagger: 0.12 }, '-=0.2')
      .fromTo('.hero-sub', { opacity: 0 }, { opacity: 1, duration: 0.5 }, '-=0.2')
      .fromTo(formRef.current, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, '-=0.1')
  }, [])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) setFile(f)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)
    if (!file) { setError('Please select a resume file.'); return }

    const fd = new FormData()
    fd.append('resume', file)

    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/upload-resume`, { method: 'POST', body: fd })
      if (!res.ok) {
        const t = await res.text()
        throw new Error(t || res.statusText)
      }
      const data = await res.json()
      const pr = data.parsedResume || data.parsed_resume || data
      setResult({
        parsed_resume: pr.parsed_resume || null,
        ats_check: pr.ats_check || null,
        content_quality: pr.content_quality || null,
        format_structure: pr.format_structure || null,
        skill_gap: pr.skill_gap || null,
      })
      setTimeout(() => {
        document.querySelector('.results-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  const scores = result ? [
    { label: 'ATS', score: result.ats_check?.score, color: '#f59e0b' },
    { label: 'Content', score: result.content_quality?.score, color: '#34d399' },
    { label: 'Format', score: result.format_structure?.score, color: '#60a5fa' },
    { label: 'Skills', score: result.skill_gap?.score, color: '#f472b6' },
  ] : []

  /* Build job search query from inferred role + skills */
  const jobQuery = result
    ? [
        result.skill_gap?.inferred_target_role,
        result.parsed_resume?.skills?.slice(0, 2).join(' '),
      ].filter(Boolean).join(' ') || 'software engineer'
    : null

  return (
    <div className="app">
      <div className="noise-overlay" aria-hidden="true" />

      {/* ── header ── */}
      <header className="site-header">
        <div className="logo">
          <span className="logo-bracket">[</span>
          <span className="logo-text">ResumeAI</span>
          <span className="logo-bracket">]</span>
        </div>
        <nav className="site-nav">
          <a href="#upload">Analyze</a>
          <a href="#results">Results</a>
          <a href="#jobs">Jobs</a>
        </nav>
      </header>

      {/* ── hero ── */}
      <section className="hero" ref={heroRef} id="upload">
        <div className="hero-inner">
          <p className="hero-eyebrow">AI-Powered Resume Intelligence</p>
          <h1 className="hero-title">
            <span>Analyze.</span>
            <span className="accent-word">Optimize.</span>
            <span>Get Hired.</span>
          </h1>
          <p className="hero-sub">
            Upload your resume and get deep ATS scoring, content analysis,<br />
            format review, skill gap insights — and live job matches.
          </p>

          <form onSubmit={handleSubmit} ref={formRef} className="upload-form">
            <div
              className={`dropzone ${dragging ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.webp"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                style={{ display: 'none' }}
              />
              {file ? (
                <div className="file-chosen">
                  <span className="file-icon">◈</span>
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{(file.size / 1024).toFixed(0)} KB</span>
                </div>
              ) : (
                <div className="drop-prompt">
                  <span className="drop-icon">⊕</span>
                  <span className="drop-main">Drop your resume here</span>
                  <span className="drop-sub">PDF, DOCX, or image — click to browse</span>
                </div>
              )}
            </div>

            <button type="submit" className="analyze-btn" disabled={loading || !file}>
              {loading ? (
                <span className="btn-loading">
                  <span className="spinner" />
                  Analyzing...
                </span>
              ) : (
                <span>Analyze Resume →</span>
              )}
            </button>
          </form>

          {error && <div className="error-bar">{error}</div>}
        </div>

        <div className="hero-grid" aria-hidden="true">
          {[...Array(6)].map((_, i) => <div key={i} className="grid-line" />)}
        </div>
      </section>

      {/* ── results ── */}
      {result && (
        <section className="results-section" id="results">
          <div className="results-inner">

            <div className="score-overview">
              <div className="score-overview-label">Overall Scores</div>
              <div className="score-rings">
                {scores.map((s) => (
                  <ScoreRing key={s.label} score={s.score} label={s.label} color={s.color} />
                ))}
              </div>
            </div>

            <ParsedResumePanel data={result.parsed_resume} />

            <AnalysisCard title="ATS Compatibility" icon="◎" accent="#f59e0b">
              <p className="card-feedback">{result.ats_check?.raw_feedback}</p>
              <div className="two-col">
                <div>
                  <p className="col-label">Found Keywords</p>
                  <TagList items={result.ats_check?.found_keywords?.slice(0, 16)} color="#f59e0b" />
                </div>
                <div>
                  <p className="col-label">Missing Keywords</p>
                  <TagList items={result.ats_check?.missing_keywords} color="#ef4444" />
                </div>
              </div>
              <div className="issues-block">
                <p className="col-label">Issues</p>
                {result.ats_check?.ats_issues?.map((issue, i) => (
                  <div key={i} className="issue-item">⚠ {issue}</div>
                ))}
              </div>
              <RecommendList items={result.ats_check?.recommendations} />
            </AnalysisCard>

            <AnalysisCard title="Content Quality" icon="✦" accent="#34d399">
              <p className="card-feedback">{result.content_quality?.raw_feedback}</p>
              <div className="mini-scores">
                {[
                  { label: 'Bullets', val: result.content_quality?.bullet_quality_score },
                  { label: 'Summary', val: result.content_quality?.summary_score },
                  { label: 'Impact', val: result.content_quality?.impact_score },
                ].map(({ label, val }) => (
                  <div key={label} className="mini-score-item">
                    <div className="mini-bar-wrap">
                      <div className="mini-bar" style={{ width: `${val || 0}%`, background: '#34d399' }} />
                    </div>
                    <span className="mini-score-label">{label}</span>
                    <span className="mini-score-val">{Math.round(val || 0)}</span>
                  </div>
                ))}
              </div>
              <div className="two-col" style={{ marginTop: '1.5rem' }}>
                <div>
                  <p className="col-label">Strong Bullets</p>
                  {result.content_quality?.strong_bullets?.map((b, i) => (
                    <div key={i} className="bullet-item strong">↑ {b}</div>
                  ))}
                </div>
                <div>
                  <p className="col-label">Weak Bullets</p>
                  {result.content_quality?.weak_bullets?.map((b, i) => (
                    <div key={i} className="bullet-item weak">↓ {b}</div>
                  ))}
                </div>
              </div>
              <RecommendList items={result.content_quality?.recommendations} />
            </AnalysisCard>

            <AnalysisCard title="Format & Structure" icon="⊞" accent="#60a5fa">
              <p className="card-feedback">{result.format_structure?.raw_feedback}</p>
              <div className="two-col">
                <div>
                  <p className="col-label">Present Sections</p>
                  <TagList items={result.format_structure?.present_sections} color="#60a5fa" />
                </div>
                <div>
                  <p className="col-label">Missing Sections</p>
                  <TagList items={result.format_structure?.missing_sections} color="#ef4444" />
                </div>
              </div>
              <div className="issues-block" style={{ marginTop: '1.5rem' }}>
                <p className="col-label">Structural Issues</p>
                {result.format_structure?.issues?.map((issue, i) => (
                  <div key={i} className="issue-item">⚠ {issue}</div>
                ))}
              </div>
              <RecommendList items={result.format_structure?.recommendations} />
            </AnalysisCard>

            <AnalysisCard title="Skill Gap Analysis" icon="◈" accent="#f472b6">
              <p className="card-feedback">{result.skill_gap?.raw_feedback}</p>
              {result.skill_gap?.inferred_target_role && (
                <div className="target-role">
                  <span className="role-label">Inferred Role</span>
                  <span className="role-val">{result.skill_gap.inferred_target_role}</span>
                </div>
              )}
              <div className="two-col" style={{ marginTop: '1.5rem' }}>
                <div>
                  <p className="col-label">Matched Skills</p>
                  <TagList items={result.skill_gap?.matched_skills} color="#34d399" />
                </div>
                <div>
                  <p className="col-label">Critical Gaps</p>
                  <TagList items={result.skill_gap?.missing_critical_skills} color="#ef4444" />
                </div>
              </div>
              {result.skill_gap?.missing_nice_to_have?.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                  <p className="col-label">Nice to Have</p>
                  <TagList items={result.skill_gap.missing_nice_to_have} color="#f59e0b" />
                </div>
              )}
              {result.skill_gap?.outdated_skills?.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                  <p className="col-label">Outdated Skills</p>
                  <TagList items={result.skill_gap.outdated_skills} color="#6b7280" />
                </div>
              )}
              <RecommendList items={result.skill_gap?.recommendations} />
            </AnalysisCard>

          </div>
        </section>
      )}

      {/* ── job results ── */}
      {result && jobQuery && (
        <JobsSection
          query={jobQuery}
          inferredRole={result.skill_gap?.inferred_target_role}
        />
      )}

      <footer className="site-footer">
        <span>ResumeAI | All rights reserved</span>
      </footer>
    </div>
  )
}