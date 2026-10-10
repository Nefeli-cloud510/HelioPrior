import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { resetWorkflow } from '../api/liveWorkflow'
import { resetClientWorkspaceState } from '../utils/missionControllerPersistence'

export function StartPage() {
  const navigate = useNavigate()
  const [resetState, setResetState] = useState<'resetting' | 'ready' | 'error'>('resetting')
  const [resetMessage, setResetMessage] = useState('正在清空上次运行记录并准备新的真实会话。')
  const [bootStep, setBootStep] = useState(0)

  useEffect(() => {
    const timers = [
      window.setTimeout(() => setBootStep(1), 420),
      window.setTimeout(() => setBootStep(2), 840),
      window.setTimeout(() => setBootStep(3), 1260),
    ]
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [])

  useEffect(() => {
    let active = true

    void (async () => {
      try {
        try {
          await resetWorkflow()
        } catch {
          // Start page should still clear local workspace state when backend service is offline.
        }
        await resetClientWorkspaceState()
        if (!active) {
          return
        }
        setResetState('ready')
        setResetMessage('当前会话已重置，可以从零开始录入科学问题并启动真实闭环。')
      } catch (error) {
        if (!active) {
          return
        }
        setResetState('error')
        setResetMessage(error instanceof Error ? error.message : '会话重置失败，请检查真实运行服务。')
      }
    })()

    return () => {
      active = false
    }
  }, [])

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--entry">
        <section className="entry-screen">
          <div className="entry-launch-beam" aria-hidden="true" />
          <div className="entry-screen__brand">
            <span className="timeline-header__eyebrow">HelioPrior</span>
            <h1>开启新的科研之旅</h1>
            <p>从中央控制智能体入口录入科学问题、上传知识材料与实验数据，再进入主流程工作台推进真实运行。</p>
            <ol className="entry-launch__steps" aria-label="启动状态">
              <li className={`entry-launch__step ${bootStep >= 1 ? 'entry-launch__step--done' : bootStep === 0 ? 'entry-launch__step--active' : ''}`}>
                你好，我是先羲 HelioPrior
              </li>
              <li className={`entry-launch__step ${bootStep >= 2 ? 'entry-launch__step--done' : bootStep === 1 ? 'entry-launch__step--active' : ''}`}>
                某处，有不可思议的事物等待被发现
              </li>
              <li className={`entry-launch__step ${bootStep >= 3 ? 'entry-launch__step--done' : bootStep === 2 ? 'entry-launch__step--active' : ''}`}>
                请说出你的科学问题，让我们一起探索
              </li>
            </ol>
            <div className={`entry-launch__note${resetState === 'error' ? ' entry-launch__note--error' : ''}`}>
              {resetMessage}
            </div>
          </div>

          <div className={`entry-screen__actions${bootStep >= 3 ? ' entry-screen__actions--ready' : ''}`}>
            <button
              type="button"
              className="entry-screen__button entry-screen__button--primary"
              onClick={() => navigate('/research')}
            >
              进入研究工作台
            </button>
            <button
              type="button"
              className="entry-screen__button"
              onClick={() => navigate('/workflow')}
            >
              打开主工作台
            </button>
            <button
              type="button"
              className="entry-screen__button"
              onClick={() => navigate('/dialogue')}
            >
              进入中央控制页
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
