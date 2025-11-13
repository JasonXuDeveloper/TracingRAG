"""Security middleware and utilities for API"""

import os
import time
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


# ============================================================================
# Password Utilities
# ============================================================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


# ============================================================================
# JWT Token Utilities
# ============================================================================


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT token

    Args:
        token: JWT token

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """Get current user from JWT token

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User data from token

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "roles": payload.get("roles", []),
    }


async def get_current_active_user(
    current_user: dict = Security(get_current_user),
) -> dict:
    """Get current active user

    Args:
        current_user: Current user from JWT

    Returns:
        Active user data

    Raises:
        HTTPException: If user is inactive
    """
    # In a real application, you would check if user is active in database
    # For now, we assume all authenticated users are active
    return current_user


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute per user
        """
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed

        Args:
            identifier: User or IP identifier

        Returns:
            True if request is allowed
        """
        now = time.time()
        minute_ago = now - 60

        # Get requests in last minute
        if identifier not in self.requests:
            self.requests[identifier] = []

        # Remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] if req_time > minute_ago
        ]

        # Check if under limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier

        Args:
            identifier: User or IP identifier

        Returns:
            Number of remaining requests
        """
        now = time.time()
        minute_ago = now - 60

        if identifier not in self.requests:
            return self.requests_per_minute

        recent_requests = [
            req_time for req_time in self.requests[identifier] if req_time > minute_ago
        ]
        return max(0, self.requests_per_minute - len(recent_requests))


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(request: Request):
    """Check rate limit for request

    Args:
        request: FastAPI request

    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get identifier (user ID or IP address)
    user_id = getattr(request.state, "user_id", None)
    identifier = user_id or request.client.host

    if not rate_limiter.is_allowed(identifier):
        remaining = rate_limiter.get_remaining(identifier)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={
                "X-RateLimit-Limit": str(rate_limiter.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "Retry-After": "60",
            },
        )


# ============================================================================
# API Key Authentication (Alternative to JWT)
# ============================================================================


API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()


async def verify_api_key(api_key: str = Security(HTTPBearer())) -> str:
    """Verify API key

    Args:
        api_key: API key from header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is invalid
    """
    if not API_KEYS:
        # If no API keys configured, allow all requests (development mode)
        return "dev-key"

    if api_key.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key.credentials


# ============================================================================
# Input Validation
# ============================================================================


def sanitize_string(input_string: str, max_length: int = 10000) -> str:
    """Sanitize user input string

    Args:
        input_string: User input
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        HTTPException: If input is too long
    """
    if len(input_string) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input exceeds maximum length of {max_length} characters",
        )

    # Remove null bytes
    sanitized = input_string.replace("\x00", "")

    return sanitized


def validate_topic_name(topic: str) -> str:
    """Validate topic name

    Args:
        topic: Topic name

    Returns:
        Validated topic name

    Raises:
        HTTPException: If topic name is invalid
    """
    if not topic or not topic.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic name cannot be empty",
        )

    # Check for SQL injection patterns
    dangerous_patterns = ["--", ";", "/*", "*/", "xp_", "sp_", "DROP", "DELETE"]
    topic_upper = topic.upper()

    for pattern in dangerous_patterns:
        if pattern in topic_upper:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Topic name contains invalid characters",
            )

    return sanitize_string(topic, max_length=255)


def validate_query(query: str) -> str:
    """Validate user query

    Args:
        query: User query

    Returns:
        Validated query

    Raises:
        HTTPException: If query is invalid
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    return sanitize_string(query, max_length=5000)


# ============================================================================
# Security Headers Middleware
# ============================================================================


async def add_security_headers(request: Request, call_next):
    """Add security headers to response

    Args:
        request: Request
        call_next: Next middleware

    Returns:
        Response with security headers
    """
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response
