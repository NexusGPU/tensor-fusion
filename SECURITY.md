# Security Policy

## Supported Versions

We actively support the following versions of Tensor Fusion with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Tensor Fusion seriously. If you believe you have found a security vulnerability in our GPU virtualization and pooling solution, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@tensor-fusion.com**

If you prefer, you can also contact us through our support channel: **support@tensor-fusion.com**

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **Initial Response**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will provide an initial assessment within 5 business days.
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days, and other vulnerabilities within 90 days.

### Responsible Disclosure

We kindly ask that you:

- Give us reasonable time to investigate and mitigate an issue before making any information public
- Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of our services
- Only interact with accounts you own or with explicit permission of the account holder

## Security Considerations

### GPU Virtualization Security

Tensor Fusion handles sensitive GPU resources and workloads. Key security considerations include:

- **Isolation**: GPU workloads are isolated between different tenants
- **Authentication**: All API access requires proper authentication
- **Authorization**: Role-based access control (RBAC) for different operations
- **Encryption**: Data in transit is encrypted using TLS 1.3
- **Audit Logging**: All administrative actions are logged for security monitoring

### Kubernetes Security

When deploying Tensor Fusion in Kubernetes:

- Use proper RBAC configurations
- Ensure network policies are in place
- Keep Kubernetes cluster updated
- Use secure container images
- Implement pod security standards

### Enterprise Features Security

Our Enterprise features include additional security measures:

- **Encryption at Rest**: GPU context and model data encryption
- **SSO/SAML Support**: Integration with enterprise identity providers
- **Advanced Audit**: Comprehensive audit trails
- **Compliance**: SOC2 and other compliance reports available

## Security Updates

Security updates will be released as patch versions and communicated through:

- GitHub Security Advisories
- Release notes
- Email notifications to enterprise customers
- Discord announcements: https://discord.gg/2bybv9yQNk

## Best Practices

### For Administrators

- Regularly update Tensor Fusion to the latest version
- Monitor security advisories and apply patches promptly
- Use strong authentication mechanisms
- Implement proper network segmentation
- Regular security audits of your deployment

### For Developers

- Follow secure coding practices when contributing
- Run security scans on your code changes
- Report any suspicious behavior or potential vulnerabilities
- Keep dependencies updated

## Security Resources

- [Apache 2.0 License](LICENSE)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Documentation](https://tensor-fusion.ai)
- [Discord Community](https://discord.gg/2bybv9yQNk)

## Contact

For security-related questions or concerns:

- **Security Team**: security@tensor-fusion.com
- **General Support**: support@tensor-fusion.com
- **Enterprise Support**: Available for licensed users

---

**Note**: This security policy applies to the open-source components of Tensor Fusion. Enterprise features may have additional security policies and procedures. Please contact us for enterprise-specific security information.
