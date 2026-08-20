# AReno Community Meeting #1

Date: 2026-08-07
Time: 14:00-15:00 China Time (UTC+8)
Participants: 18
Format: Online community meeting

## Summary

The first AReno community meeting was held successfully with 18 participants.
The meeting focused on recent release updates, upcoming multimodal work,
installation experience improvements, contributor support, and community
collaboration norms.

This was also the first regular community sync. Going forward, meeting notes
and discussion topics will be synchronized to GitHub so contributors can review
past discussions, propose agenda items, and follow up through issues.

## Topics

### 1. Community Meeting Format

- The community meeting will be used as a regular space for project updates,
  contributor discussion, Q&A, and issue follow-up.
- Future meeting topics should be collected through GitHub issues before the
  meeting.
- Participants are encouraged to raise questions in the group chat, GitHub
  issues, or directly during the meeting.

### 2. AReno 0.0.7 Release Preview

The upcoming `0.0.7` release was introduced with three major areas:

- Support for different rollout/training topologies and full exchange.
- Initial multimodal support, including image input support for Qwen3.5 and
  Qwen3.5-MoE.
- Dashboard refactor, mainly improving interaction and observability
  experience.

Additional small bug fixes are also expected in this release.

### 3. Multimodal Roadmap

Multimodal capability was discussed as one of the most important upcoming
directions for AReno.

Key points:

- Short-term work should focus on multimodal understanding first, especially
  image, video, and audio inputs.
- Full omni-model support, including any-to-any input/output, is considered
  lower priority for the short term.
- Gemma was discussed as a possible first model family to validate video, audio,
  and image input support because it has relatively complete multimodal coverage.
- Generation of images, audio, or video is more complex and should be scoped
  carefully after input-side understanding works.

### 4. Installation Experience

Community members shared installation problems, especially around WSL2, Docker,
FlashAttention, and long dependency build times.

Discussion points:

- Docker images exist, but the end-to-end first-run experience still needs to be
  improved.
- Installing FlashAttention from source can be slow; prebuilt wheels may improve
  the experience for common environments.
- AI-assisted installation could become a better path for new users if supported
  with clearer commands, skills, or setup guidance.
- Installation documentation should preserve richer media such as images or
  videos where possible, because text-only docs can lose important context.

### 5. Training, Checkpoints, and GPU Resources

Several user questions focused on how to continue using models after training
and how to avoid resource issues during training.

Notes:

- Trained checkpoints can be downloaded and served locally if the user has the
  checkpoint files and a compatible serving setup.
- For out-of-memory or unstable training runs, users can reduce rollout memory
  usage, lower mini-batch size, or adjust tensor-parallel settings.
- Users should avoid pushing GPU memory usage to the limit, because near-limit
  runs are more likely to fail.
- Google Colab and Google Cloud trial credits were mentioned as possible GPU
  resource options for students.

### 6. Contributor Support and Community Operations

- The community will continue encouraging students and contributors to ask
  questions early, including beginner questions.
- Good contributors may receive community gifts such as bags, sun-protective
  clothing, and stickers.
- Contributors who write AReno-related articles or experiment reports can get
  help with editing, publishing, and distribution.
- Recent PR contributors and documentation contributors will be considered for
  community recognition.

## Decisions

- Meeting notes and discussion outcomes will be synchronized to GitHub.
- Future community meetings should have a GitHub issue opened about one week in
  advance for agenda collection.
- The next community meeting will be held on 2026-08-21 from 13:00 to 14:00
  China Time.
- Multimodal input understanding should be prioritized before broader
  omni-model or generation support.
- Installation bad cases should be collected from users and turned into
  actionable improvements.

## Action Items

- Open a GitHub issue before the next meeting to collect agenda items.
- Create follow-up issues for multimodal model adaptation, starting with image,
  video, and audio input support.
- Investigate whether prebuilt wheels can cover several common installation
  environments.
- Collect WSL2, Docker, and FlashAttention installation bad cases from users.
- Improve installation docs with clearer steps, richer media, and links to
  relevant tutorial videos.
- Discuss opt-in user usage data collection for future versions, including what
  data is collected and how users consent.
- Encourage users to share training errors, installation screenshots, and
  environment details in the community group or GitHub issues.

## Open Questions

- Which environments should be prioritized for prebuilt wheel coverage?
- Should AReno provide official guidance for uploading checkpoints to platforms
  such as Hugging Face or ModelScope, or link to upstream tutorials instead?
- What should the first AI-native installation workflow look like?
- What telemetry, if any, is useful enough to collect with explicit user opt-in?

## Next Meeting

Date: 2026-08-21
Time: 13:00-14:00 China Time (UTC+8)
Suggested agenda:

- Review installation bad cases collected from users.
- Follow up on multimodal input support progress.
- Discuss candidate issues for new contributors.
- Continue open Q&A for training, serving, and environment setup.

## Links

- Recording:
- Transcript:
- Slides:
- Related issues:
