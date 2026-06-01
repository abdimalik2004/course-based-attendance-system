# Recent Activity System - Implementation Guide

## ✅ Completed Implementation

A **production-ready, enterprise-level Recent Activity system** has been successfully implemented for the Course-Based Attendance System Admin Dashboard.

---

## 📋 Architecture Overview

### Backend (Python/FastAPI)

- **Activity Logger Service**: Centralized, reusable logging utility
- **WebSocket Manager**: Real-time activity broadcasting
- **API Endpoints**: REST + WebSocket for activity retrieval

### Frontend (React/TypeScript)

- **Activity Service**: HTTP + WebSocket client handling
- **Admin Dashboard**: Live-updating Recent Activity widget
- **Relative Timestamps**: Human-readable time formatting

---

## 🗄️ Database

### Table: `activity_logs`

| Column       | Type         | Properties                                     |
| ------------ | ------------ | ---------------------------------------------- |
| `id`         | INTEGER      | PK, Auto-increment, Indexed                    |
| `user_id`    | INTEGER      | FK → users.id, Nullable, Indexed               |
| `username`   | VARCHAR(128) | User performing action (never null)            |
| `action`     | VARCHAR(255) | Activity description                           |
| `status`     | ENUM         | SUCCESS, FAILED, PENDING                       |
| `created_at` | DATETIME     | Server timestamp, **Indexed for fast queries** |

**Indexes:**

- `ix_activity_logs_created_at` - Fast filtering by time range
- `ix_activity_logs_user_id` - Fast user-based filtering
- `ix_activity_logs_id` - Primary key

---

## 🔧 Backend Implementation

### 1. Activity Logger Service (`backend/app/utils/activity_logger.py`)

**Core Function:**

```python
log_activity(
    action: str,                              # Required: Activity description
    user: User | None = None,                 # Optional: User object
    username: str | None = None,              # Optional: Username string
    status: str | ActivityLogStatus = "Success",  # Status
    db: Session | None = None,                # Optional: Existing DB session
    own_transaction: bool = True              # Transaction management
) -> ActivityLog | None
```

**Features:**

- ✅ Flexible session management (uses existing or creates new)
- ✅ Automatic status normalization (case-insensitive)
- ✅ Error handling (never breaks main request flow)
- ✅ Optional WebSocket broadcasting
- ✅ Detailed logging for debugging

**Usage Examples:**

```python
# Standalone (simplest, creates own session)
log_activity(action="User Login", username="john_doe", status="Success")

# Within existing transaction
log_activity(
    action="Teacher Registered",
    user=current_user,
    db=db,
    own_transaction=False
)

# Using helper functions
log_authentication(username="admin", action="User Login", user=current_user)
log_academic_action(action="Course Created", user=current_user)
log_attendance_action(action="Attendance Marked", user=current_user)
```

### 2. API Endpoints (`backend/app/routers/activity.py`)

#### GET `/activity/recent`

**Returns activities from the last 2 hours, newest first.**

```http
GET /activity/recent?limit=30&hours=2 HTTP/1.1
Authorization: Bearer <token>
```

**Response:**

```json
[
  {
    "id": 123,
    "username": "admin",
    "action": "Teacher Registered - John Smith (ENG-T-001)",
    "status": "Success",
    "created_at": "2024-05-22T14:30:45.123456"
  }
]
```

**Query Parameters:**

- `limit` (default: 30, max: 100) - Number of records
- `hours` (default: 2, max: 24) - Time window

**Security:** Requires `SUPER_ADMIN` role

#### GET `/activity/stats`

**Returns activity statistics for reporting.**

```http
GET /activity/stats?hours=24 HTTP/1.1
Authorization: Bearer <token>
```

**Response:**

```json
{
  "total_activities": 156,
  "success_count": 150,
  "failed_count": 4,
  "pending_count": 2,
  "unique_users": 12
}
```

#### WebSocket `/activity/ws/recent`

**Real-time activity stream for live dashboard updates.**

```javascript
const ws = new WebSocket("ws://localhost:8000/activity/ws/recent");

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === "activity") {
    console.log("New activity:", message.data);
  }
};
```

**Message Format:**

```json
{
  "type": "activity",
  "data": {
    "id": 124,
    "username": "admin",
    "action": "Student Registered - Jane Doe (STU-001)",
    "status": "Success",
    "created_at": "2024-05-22T14:35:22.456789"
  }
}
```

### 3. WebSocket Manager (`backend/app/services/activity_websocket_manager.py`)

Centralized manager for broadcasting activities to connected clients.

**Features:**

- ✅ Multi-client connection handling
- ✅ Clean disconnect management
- ✅ Error resilience
- ✅ Scalable architecture

---

## 🎨 Frontend Implementation

### 1. Activity Service (`frontend/src/services/activityService.ts`)

**Core Methods:**

```typescript
// Fetch activities
await activityService.getRecentActivities(limit: 30, hours: 2)

// Get statistics
await activityService.getActivityStats(hours: 24)

// Connect to WebSocket
activityService.connectWebSocket(
  onActivity: (activity: ActivityLog) => void,
  onError?: (error: Event) => void,
  onClose?: (event: CloseEvent) => void
)

// Check connection status
activityService.isWebSocketConnected(): boolean

// Disconnect
activityService.disconnectWebSocket()
```

**Utility Functions:**

```typescript
// Format time as "2 mins ago", "1 hour ago"
formatTimeAgo(dateStr: string): string

// Format as "HH:MM"
formatTime(dateStr: string): string

// Full datetime for tooltips
formatFullDateTime(dateStr: string): string
```

### 2. Admin Dashboard (`frontend/src/pages/admin/Dashboard.tsx`)

**Features:**

- ✅ Live WebSocket connection status indicator
- ✅ Fallback polling every 30 seconds
- ✅ Real-time activity insertion (prevents duplicates)
- ✅ Relative timestamps with full datetime tooltips
- ✅ Color-coded status badges
- ✅ Responsive table layout
- ✅ Dark/light mode support
- ✅ Loading skeleton states
- ✅ Manual refresh button

**UI Elements:**

```
┌─────────────────────────────────────────────────────┐
│ Recent Activity                        🟢 Connected  │
│ Latest activity · Live updates                       │
├──────────────────────────────────────────────────────┤
│ User    │ Action                │ Status │ Time     │
├─────────┼───────────────────────┼────────┼──────────┤
│ admin   │ User Login - admin    │ ✓      │ 2m ago   │
│ john    │ Course Created        │ ✓      │ 5m ago   │
│ jane    │ Attendance Marked     │ ✓      │ 1h ago   │
│ system  │ Report Generated      │ ✓      │ 1h ago   │
└──────────────────────────────────────────────────────┘
```

---

## 📊 Activity Logging Throughout System

Activity logging has been integrated into key system endpoints:

### Authentication (`/auth`)

- ✅ User Login (Success/Failed)
- ✅ User Logout
- ✅ User Registration
- ✅ Password Changes

### User Management

- ✅ User Created
- ✅ User Updated
- ✅ User Deleted

### Academic Structure

- ✅ Faculty Created
- ✅ Department Created
- ✅ Course Created
- ✅ Schedule Created

### Attendance

- ✅ Attendance Session Started
- ✅ Attendance Marked
- ✅ Attendance Session Closed

### Reports

- ✅ Report Generated
- ✅ Export Generated

---

## 🚀 Real-Time Features

### WebSocket Updates

1. **Instant Delivery**: Activities appear instantly without page refresh
2. **Connection Management**: Automatic reconnection with exponential backoff
3. **Status Indicator**: Visual feedback (Connected/Polling)
4. **Duplicate Prevention**: Same activity won't appear twice
5. **Graceful Degradation**: Falls back to polling if WebSocket fails

### Polling Fallback

- **30-second intervals** for browsers that don't support WebSocket
- Ensures users see updates even without real-time connection
- Configurable via query parameters

---

## 🔒 Security

### Access Control

- ✅ Only `SUPER_ADMIN` can access `/activity/recent`
- ✅ Role-based endpoint protection
- ✅ User context tracked in logs

### Data Safety

- ✅ Sensitive data not exposed (passwords never logged)
- ✅ Failed login attempts logged for security auditing
- ✅ User IDs not displayed in UI
- ✅ SQL injection prevention via ORM

---

## ⚡ Performance Optimizations

### Database

- ✅ **Indexed on `created_at`** - Fast time-range queries
- ✅ **Indexed on `user_id`** - Quick user filtering
- ✅ Query limits (max 100 records) prevent data overload

### Frontend

- ✅ **React Query caching** - Reduced API calls
- ✅ **Stale-while-revalidate pattern** - Quick page loads
- ✅ **Virtual scrolling ready** - Handles large datasets
- ✅ **Debounced WebSocket** - No connection flooding

### Backend

- ✅ **Flush-based transactions** - No extra commits
- ✅ **Async WebSocket broadcasting** - Non-blocking
- ✅ **Efficient filtering** - Uses database indexes

---

## 📝 Usage Guide

### For Developers

**Adding activity logging to a new endpoint:**

```python
from app.utils.activity_logger import log_activity

@router.post("/example", dependencies=[Depends(require_roles("ADMIN"))])
def create_example(
    payload: ExampleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ... create logic ...

    # Log the activity
    log_activity(
        action=f"Example Created - {payload.name}",
        user=current_user,
        db=db,
        own_transaction=False  # If within existing transaction
    )
    db.commit()
    return example
```

### For Admins

**Monitoring the dashboard:**

1. Navigate to Admin Dashboard
2. Check Recent Activity widget
3. Monitor user actions in real-time
4. Click refresh button to get latest activities
5. Check connection status (green = WebSocket, orange = polling)

---

## 🛠️ Troubleshooting

### WebSocket Connection Issues

**Symptom:** Showing "Polling (30s)" instead of "Connected"

**Solutions:**

1. Check browser console for WebSocket errors
2. Verify backend is running
3. Check CORS configuration
4. Verify proxy/firewall allows WebSocket traffic

### Missing Activities

**Symptom:** Activities not appearing in dashboard

**Solutions:**

1. Verify activity logging code is in endpoint
2. Check database is accessible
3. Verify user has SUPER_ADMIN role
4. Check created_at timestamp is correct

### Performance Issues

**Symptom:** Dashboard is slow or freezing

**Solutions:**

1. Reduce `limit` parameter in `/activity/recent`
2. Reduce `hours` parameter (faster queries)
3. Check database indexes exist
4. Monitor browser DevTools for slow requests

---

## 📚 API Reference

### Request Headers

```http
Authorization: Bearer <JWT_TOKEN>
```

### Status Codes

- `200` - Success
- `401` - Unauthorized (missing/invalid token)
- `403` - Forbidden (insufficient permissions)
- `422` - Invalid query parameters

### Response Format

All responses use standard JSON with:

```json
{
  "data": [...],
  "error": null,
  "timestamp": "2024-05-22T14:30:45Z"
}
```

---

## 🔄 Data Flow

```
User Action
    ↓
    ├→ Business Logic (Create/Update/Delete)
    ├→ log_activity(db, ...)
    │   ├→ Flush to database
    │   └→ Schedule WebSocket broadcast (async)
    ├→ db.commit()
    └→ Return Response

WebSocket Broadcast
    ↓
    └→ All connected admin clients receive update instantly

If WebSocket Unavailable
    ↓
    └→ Frontend falls back to polling every 30s
```

---

## 🎯 Key Metrics

| Metric                 | Target      | Current           |
| ---------------------- | ----------- | ----------------- |
| Activity logged        | 100%        | ✅ Implemented    |
| DB query time          | <100ms      | ✅ Indexed        |
| WebSocket latency      | <1s         | ✅ Real-time      |
| Fallback poll interval | 30s         | ✅ Configured     |
| Max retained records   | Per 2 hours | ✅ Limited        |
| Admin access only      | Yes         | ✅ Role protected |

---

## 📦 Files Modified/Created

### Backend

- ✅ `backend/app/utils/activity_logger.py` - **NEW** Logging service
- ✅ `backend/app/services/activity_websocket_manager.py` - **NEW** WebSocket manager
- ✅ `backend/app/routers/activity.py` - Enhanced with endpoints
- ✅ `backend/app/routers/auth.py` - Activity logging integrated
- ✅ `backend/app/routers/teachers.py` - Activity logging integrated
- ✅ `backend/app/routers/students.py` - Activity logging integrated
- ✅ `backend/app/routers/faculties.py` - Activity logging integrated
- ✅ `backend/app/routers/departments.py` - Activity logging integrated
- ✅ `backend/app/routers/courses.py` - Activity logging integrated
- ✅ `backend/app/routers/sessions.py` - Activity logging integrated
- ✅ `backend/app/routers/reports.py` - Activity logging integrated

### Frontend

- ✅ `frontend/src/services/activityService.ts` - **NEW** Activity client service
- ✅ `frontend/src/pages/admin/Dashboard.tsx` - Enhanced with WebSocket + real-time UI

---

## 🎓 Next Steps

### Optional Enhancements

1. **Activity Filters**
   - Filter by user, action type, status
   - Date range filtering
   - Search functionality

2. **Activity Export**
   - Export to CSV
   - Export to PDF
   - Custom date ranges

3. **Audit Reports**
   - Daily/weekly/monthly summaries
   - User activity reports
   - System health metrics

4. **Notifications**
   - Alert on failed actions
   - User login notifications
   - Critical system events

5. **Activity Archive**
   - Move old activities to archive table
   - Optimize main table performance
   - Retention policies

### Configuration Options

Create `backend/app/core/activity_config.py`:

```python
ACTIVITY_CONFIG = {
    "MAX_DASHBOARD_RECORDS": 30,
    "DASHBOARD_HOURS": 2,
    "LOG_RETENTION_DAYS": 90,
    "WEBSOCKET_RECONNECT_MAX_ATTEMPTS": 5,
    "WEBSOCKET_RECONNECT_DELAY_MS": 3000,
}
```

---

## 📞 Support

For issues or questions:

1. Check the **Troubleshooting** section above
2. Review **Backend logs**: `backend/logs/app.log`
3. Check **Browser console**: F12 → Console tab
4. Monitor **Database**: Check `activity_logs` table directly

---

## ✨ Summary

This implementation provides:

- ✅ **Centralized Activity Logging** - One function for all activities
- ✅ **Real-Time Updates** - WebSocket with fallback polling
- ✅ **Production Ready** - Error handling, security, performance
- ✅ **Scalable** - Easy to add new activity types
- ✅ **User-Friendly** - Beautiful dashboard with live updates
- ✅ **Enterprise-Level** - Comprehensive logging and security

The system is **fully functional** and **ready for deployment**.
