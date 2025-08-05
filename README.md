<p align="center"><a href="javascript:void(0);" target="_blank" rel="noreferrer"><img width="100%" src="https://cdn.tensor-fusion.ai/logo-banner.png" alt="Logo"></a></p>

<p align="center">
    <br /><strong><a href="https://tensor-fusion.ai" target="_blank">TensorFusion.AI</a></strong><br/><b>Less GPUs, More AI Apps.</b>
    <br />
    <a href="https://tensor-fusion.ai/guide/overview"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://tensor-fusion.ai/guide/overview">View Demo</a>
    |
    <a href="https://github.com/NexusGPU/tensor-fusion/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    |
    <a href="https://github.com/NexusGPU/tensor-fusion/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NexusGPU/tensor-fusion)

Tensor Fusion is a state-of-the-art **GPU virtualization and pooling solution** designed to optimize GPU cluster utilization to its fullest potential.

## 🌟 Highlights

#### 📐 Fractional Virtual GPU
#### 🔄 Remote GPU Sharing over Ethernet/InfiniBand
#### ⚖️ GPU-first Scheduling and Auto-scaling
#### 📊 GPU Oversubscription and VRAM Expansion
#### 🛫 GPU Pooling, Monitoring, Live Migration, Model Preloading and more

## 🎬 Demo

### Fractional vGPU & GPU-over-IP & Distributed Allocation

![Fractional vGPU & GPU-over-IP & Distributed Allocation](https://cdn.tensor-fusion.ai//demo/overall-demo.gif)


### AI Infra Console

![AI Infra Console](https://cdn.tensor-fusion.ai//demo/ai-infra-console.gif)

### GPU Live-migration [End-to-end feature WIP]

https://cdn.tensor-fusion.ai/GPU_Content_Migration.mp4

## 🚀 Quick Start

### Onboard Your Own AI Infra

- [Deploy in Kubernetes cluster](https://tensor-fusion.ai/guide/getting-started/deployment-k8s)
- [Create new cluster in VM/BareMetal](https://tensor-fusion.ai/guide/getting-started/deployment-vm)
- [Learn Essential Concepts & Architecture](https://tensor-fusion.ai/guide/getting-started/architecture)

<!-- (TODO: Asciinema) -->

<!-- ### Playground

- Explore the demo account: [Demo Console - Working in progress](https://app.tensor-fusion.ai?hint=demo) -->

### 💬 Discussion

- Discord channel: [https://discord.gg/2bybv9yQNk](https://discord.gg/2bybv9yQNk)
- Discuss anything about TensorFusion: [Github Discussions](https://github.com/NexusGPU/tensor-fusion/discussions)
- Contact us with WeCom for Greater China region: [企业微信](https://work.weixin.qq.com/ca/cawcde42751d9f6a29) 
- Email us: [support@tensor-fusion.com](mailto:support@tensor-fusion.com)
- Schedule [1:1 meeting with TensorFusion founders](https://tensor-fusion.ai/book-demo)


## 🔮 Features & Roadmap

### Core GPU Virtualization Features

- [x] Fractional GPU and flexible oversubscription
- [x] Remote GPU sharing with SOTA GPU-over-IP technology, less than 4% performance loss
- [x] GPU VRAM expansion and hot/warm/cold tiering
- [ ] None NVIDIA GPU/NPU vendor support

### Pooling & Scheduling & Management

- [x] GPU/NPU pool management in Kubernetes
- [x] GPU-first scheduling and allocation, with single TFlops/MB precision
- [x] GPU node auto provisioning/termination
- [x] GPU compaction/bin-packing
- [x] Seamless onboarding experience for Pytorch, TensorFlow, llama.cpp, vLLM, Tensor-RT, SGlang and all popular AI training/serving frameworks
- [x] Centralized Dashboard & Control Plane
- [x] GPU-first autoscaling policies, auto set requests/limits/replicas
- [x] Request multiple vGPUs with group scheduling for large models
- [x] Support different QoS levels

### Enterprise Features

- [x] GPU live-migration, snapshot and restore GPU context cross cluster
- [ ] AI model registry and preloading, build your own private MaaS(Model-as-a-Service)
- [ ] Advanced auto-scaling policies, scale to zero, rebalance of hot GPUs
- [ ] Advanced observability features, detailed metrics & tracing/profiling of CUDA calls
- [ ] Monetize your GPU cluster by multi-tenancy usage measurement & billing report
- [ ] Enterprise level high availability and resilience, support topology aware scheduling, GPU node auto failover etc.
- [ ] Enterprise level security, complete on-premise deployment support
- [ ] Enterprise level compliance, SSO/SAML support, advanced audit, ReBAC control, SOC2 and other compliance reports available

### 🗳️ Platform Support

- [x] Run on Linux Kubernetes clusters
- [x] Run on Linux VMs or Bare Metal (one-click onboarding to Edge K3S)
- [x] Run on Windows (Not open sourced, contact us for support)

See the [open issues](https://github.com/NexusGPU/tensor-fusion/issues) for a full list of proposed features (and known issues).

## 🙏 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors

<a href="https://github.com/NexusGPU/tensor-fusion/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NexusGPU/tensor-fusion" alt="contrib.rocks image" />
</a>

## 🔷 License

1. [TensorFusion main repo](https://github.com/NexusGPU/tensor-fusion) is open sourced with [Apache 2.0 License](./LICENSE), which includes **GPU pooling, scheduling, management features**, you can use it for free and customize it as you want.
2. [vgpu.rs repo](https://github.com/NexusGPU/vgpu.rs) is open sourced with [Apache 2.0 License](./LICENSE), which includes **Fractional GPU** and **vGPU hypervisor features**, you can use it for free and customize it as you want.
3. **Advanced GPU virtualization and GPU-over-IP sharing features** are also free to use when **GPU total number of your organization is less than 10**, but the implementation is not fully open sourced, please [contact us](mailto:support@tensor-fusion.com) for more details.
4. Features mentioned in "**Enterprise Features**" above are paid, **licensed users can use these features in [TensorFusion Console](https://app.tensor-fusion.ai)**.
5. For large scale deployment that involves non-free features of #3 and #4, please [contact us](mailto:support@tensor-fusion.com), pricing details are available [here](https://tensor-fusion.ai/pricing)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FNexusGPU%2Ftensor-fusion.svg?type=large&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FNexusGPU%2Ftensor-fusion?ref=badge_large&issueType=license)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/NexusGPU/tensor-fusion.svg?style=for-the-badge
[contributors-url]: https://github.com/NexusGPU/tensor-fusion/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NexusGPU/tensor-fusion.svg?style=for-the-badge
[forks-url]: https://github.com/NexusGPU/tensor-fusion/network/members
[stars-shield]: https://img.shields.io/github/stars/NexusGPU/tensor-fusion.svg?style=for-the-badge
[stars-url]: https://github.com/NexusGPU/tensor-fusion/stargazers
[issues-shield]: https://img.shields.io/github/issues/NexusGPU/tensor-fusion.svg?style=for-the-badge
[issues-url]: https://github.com/NexusGPU/tensor-fusion/issues
[license-shield]: https://img.shields.io/github/license/NexusGPU/tensor-fusion.svg?style=for-the-badge
[license-url]: https://github.com/NexusGPU/tensor-fusion/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/tensor-fusion/about

