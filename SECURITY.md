# Security Policy

## Supported Versions

This project is actively maintained and security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Vulnerabilities

### Known Vulnerabilities and Mitigations

#### 1. Gradio CVE-2024-39236 (Code Injection)
- **Status**: Mitigated
- **Description**: Code injection vulnerability via component_meta.py
- **Mitigation**: 
  - Disabled authentication features that could be exploited
  - Restricted file access with `allowed_paths=[]`
  - Limited concurrent requests with `max_threads=1`
  - Disabled error details exposure with `show_error=False`

#### 2. Gradio CVE-2025-5320 (Origin Validation)
- **Status**: Mitigated
- **Description**: Origin validation vulnerability in is_valid_origin function
- **Mitigation**:
  - Disabled queue functionality with `enable_queue=False`
  - Implemented strict input validation and sanitization
  - Added path validation for all file operations

#### 3. ONNX CVE-2024-5187 (Arbitrary File Overwrite)
- **Status**: Mitigated
- **Description**: Arbitrary file overwrite vulnerability in download_model_with_test_data
- **Mitigation**:
  - Added strict path validation for ONNX model loading
  - Implemented file existence checks before operations
  - Used secure session options with restricted capabilities
  - Disabled potentially vulnerable memory optimizations

### Security Measures Implemented

#### Input Validation and Sanitization
- All user inputs are validated and sanitized
- Type checking for all parameters
- Range validation for numeric inputs
- String sanitization to prevent injection attacks

#### File System Security
- Path validation to prevent directory traversal
- Restricted file access to specific directories
- File existence checks before operations
- No arbitrary file operations allowed

#### Network Security
- Disabled external sharing (`share=False`)
- Restricted server binding to localhost
- Disabled analytics and telemetry
- Limited concurrent connections

#### Error Handling
- Generic error messages to prevent information disclosure
- No stack traces exposed to users
- Proper exception handling throughout the codebase

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do not create a public issue** for the vulnerability
2. **Email the details** to: security@openllm.org
3. **Include the following information**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Release**: Within 30 days (depending on severity)

## Security Best Practices

### For Developers
1. Always validate and sanitize user inputs
2. Use parameterized queries and prepared statements
3. Implement proper authentication and authorization
4. Keep dependencies updated
5. Follow the principle of least privilege

### For Users
1. Keep the application updated
2. Use strong authentication
3. Monitor logs for suspicious activity
4. Report security issues promptly
5. Follow deployment security guidelines

## Dependency Security

### Regular Security Scans
- Automated security scanning with `safety`
- Bandit static analysis for Python code
- Regular dependency updates
- Vulnerability monitoring

### Security Tools Used
- `safety`: Dependency vulnerability scanning
- `bandit`: Static security analysis
- `black`: Code formatting (prevents some injection attacks)
- `isort`: Import organization (security best practice)

## Compliance

This project follows security best practices and implements mitigations for known vulnerabilities. However, no software is completely secure, and users should:

1. Regularly update dependencies
2. Monitor security advisories
3. Implement additional security measures as needed
4. Conduct regular security audits

## Contact

For security-related questions or concerns:
- Email: security@openllm.org
- GitHub Security: Use the security tab in the repository
- Issues: Create a private security issue (if available)

---

**Note**: This security policy is a living document and will be updated as new vulnerabilities are discovered and mitigated.