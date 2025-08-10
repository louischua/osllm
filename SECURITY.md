# Security Policy

## Supported Versions

We take security seriously and actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | ✅ Yes             |
| 0.2.x   | ✅ Yes (when released) |

## Reporting a Vulnerability

If you discover a security vulnerability in OpenLLM, please help us keep the project secure by following responsible disclosure:

### 🔒 **Private Reporting (Preferred)**

**Email:** [louischua@gmail.com](mailto:louischua@gmail.com?subject=OpenLLM%20Security%20Vulnerability)

**Subject:** `OpenLLM Security Vulnerability`

**Include in your report:**
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### ⏰ **Response Timeline**

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 1 week
- **Status Updates:** Weekly until resolved
- **Fix Timeline:** Depends on severity (see below)

### 🚨 **Severity Levels**

| Severity | Response Time | Description |
|----------|---------------|-------------|
| **Critical** | 24-48 hours | Remote code execution, data exposure |
| **High** | 1 week | Privilege escalation, authentication bypass |
| **Medium** | 2-4 weeks | Information disclosure, DoS |
| **Low** | 1-2 months | Minor security improvements |

### 🛡️ **Security Scope**

**In Scope:**
- Core training pipeline security
- Model inference vulnerabilities
- API endpoint security (FastAPI server)
- Data processing pipeline issues
- Dependency vulnerabilities
- Authentication/authorization flaws

**Out of Scope:**
- Training data quality or bias (see ethical AI guidelines)
- Performance optimization suggestions
- Feature requests
- General bug reports (use GitHub Issues)

### 🏆 **Recognition**

We appreciate security researchers who help keep OpenLLM secure:
- Public acknowledgment in security advisories (if desired)
- Listed in our security hall of fame
- Priority support for future security research

### 📚 **Security Best Practices**

When using OpenLLM:
- Keep dependencies updated (`pip install -U -r requirements.txt`)
- Use secure model serving configurations
- Validate all user inputs in production
- Monitor for unusual inference patterns
- Follow the security guidelines in our documentation

### 🔍 **Vulnerability Disclosure Policy**

- We request 90 days to address critical vulnerabilities before public disclosure
- We will provide security advisories for all confirmed vulnerabilities
- Coordinated disclosure is preferred to protect users

Thank you for helping keep OpenLLM and its users safe! 🛡️