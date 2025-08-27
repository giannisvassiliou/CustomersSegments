import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
import logging
import threading
import time
import queue
import json
from sklearn.metrics import silhouette_score

from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import random
import copy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for system classification
class AlertSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class EvolutionEventType(Enum):
    SEGMENT_DEATH = "SEGMENT_DEATH"
    SEGMENT_BIRTH = "SEGMENT_BIRTH"
    SEGMENT_SPLIT = "SEGMENT_SPLIT"
    SEGMENT_MERGE = "SEGMENT_MERGE"
    SEGMENT_DRIFT = "SEGMENT_DRIFT"
    SIZE_CHANGE = "SIZE_CHANGE"

@dataclass
class UserActivity:
    """Real-time user activity data"""
    user_id: str
    timestamp: datetime
    session_duration: float
    pages_viewed: int
    clicks: int
    purchases: int
    purchase_amount: float
    device_type: str
    location: str
    time_since_last_visit: float
    bounce_rate: float
    conversion_event: bool

@dataclass
class UserProfile:
    """Comprehensive user profile built from activities"""
    user_id: str
    first_seen: datetime
    last_seen: datetime
    total_sessions: int
    avg_session_duration: float
    total_pages_viewed: int
    total_clicks: int
    total_purchases: int
    total_revenue: float
    preferred_device: str
    primary_location: str
    engagement_score: float
    loyalty_score: float
    value_score: float
    behavior_vector: np.ndarray
    activity_history: List[UserActivity]

@dataclass
class SegmentData:
    """Dynamic segment information"""
    segment_id: str
    segment_name: str
    user_ids: set
    centroid: np.ndarray
    timestamp: datetime
    size: int
    behavioral_metrics: Dict[str, float]
    stability_score: float
    growth_rate: float
    cohesion_score: float
    silhouette_score: float = 0.0  # ADD THIS LINE


@dataclass

class EvolutionEvent:
    event_id: str
    event_type: EvolutionEventType
    segment_id: str
    timestamp: datetime
    confidence: float
    severity: AlertSeverity
    details: Dict[str, Any]
    predicted_impact: Dict[str, float]
    affected_users: List[str]
    # Add revenue tracking fields
    revenue_impact: float = 0.0  # Positive for gains, negative for losses
    revenue_before: float = 0.0
    revenue_after: float = 0.0
    revenue_per_user_before: float = 0.0
    revenue_per_user_after: float = 0.0
# Immediate Notification System
class ImmediateNotificationCallback:
    """Callback interface for immediate notifications"""
    
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        """Called immediately when a new segment is born"""
        pass
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        """Called immediately when a segment dies"""
        pass
    
    def on_segment_size_change(self, event: EvolutionEvent, segment: SegmentData):  # ADD THIS METHOD
        """Called immediately when a segment size changes significantly"""
        pass


class ConsoleNotificationCallback(ImmediateNotificationCallback):
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        print(f"\n*** SEGMENT BIRTH ALERT! ***")
        print(f"   Segment: {segment.segment_name}")
        print(f"   👥 NEW USERS: +{segment.size} users")
        print(f"   💰 REVENUE GAIN: ${event.revenue_impact:,.2f}")
        print(f"   💰 Revenue per User: ${event.revenue_per_user_after:.2f}")
        print(f"   Timestamp: {event.timestamp}")
        print("-" * 60)
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        print(f"\n*** SEGMENT DEATH ALERT! ***")
        print(f"   Segment: {last_known_segment.segment_name}")
        print(f"   👥 USERS LOST: -{last_known_segment.size} users")
        print(f"   💸 REVENUE LOST: ${abs(event.revenue_impact):,.2f}")
        print(f"   💸 Revenue per User Lost: ${event.revenue_per_user_before:.2f}")
        print(f"   Timestamp: {event.timestamp}")
        print("-" * 60)
    def on_segment_size_change(self, event: EvolutionEvent, segment: SegmentData):  # ADD THIS METHOD
        user_change = event.details.get('size_change', 0)
        change_type = "GROWTH" if user_change > 0 else "SHRINKAGE"
        change_symbol = "+" if user_change >= 0 else ""
        
        print(f"\n*** SEGMENT {change_type} ALERT! ***")
        print(f"   Segment: {segment.segment_name}")
        print(f"   User Change: {change_symbol}{user_change} users")
        print(f"   Size: {event.details.get('previous_size', 0)} → {event.details.get('current_size', 0)}")
        print(f"   Revenue Change: ${event.revenue_impact:,.2f}")
        print(f"   Timestamp: {event.timestamp}")
        print("-" * 60)
class EmailNotificationCallback(ImmediateNotificationCallback):
    """Email-based immediate notifications"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 to_emails: List[str], from_email: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_emails = to_emails
        self.from_email = from_email or username
    
    def _send_email(self, subject: str, body: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        subject = f"ALERT: New Customer Segment Born - {segment.segment_name}"
        body = f"""
IMMEDIATE SEGMENT BIRTH ALERT

A new customer segment has been detected in real-time:

Segment Details:
- ID: {segment.segment_id}
- Name: {segment.segment_name}
- Size: {segment.size} users
- Timestamp: {event.timestamp}
- Confidence: {event.confidence:.2f}

Behavioral Metrics:
{json.dumps(segment.behavioral_metrics, indent=2)}

Impact Analysis:
{json.dumps(event.predicted_impact, indent=2)}

Affected Users: {len(event.affected_users)}

This is an automated alert from the Real-Time Clustering System.
        """
        self._send_email(subject, body)
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        subject = f"ALERT: Customer Segment Died - {last_known_segment.segment_name}"
        body = f"""
IMMEDIATE SEGMENT DEATH ALERT

A customer segment has disappeared from the system:

Segment Details:
- ID: {last_known_segment.segment_id}
- Name: {last_known_segment.segment_name}
- Last Size: {last_known_segment.size} users
- Death Timestamp: {event.timestamp}
- Confidence: {event.confidence:.2f}

Last Known Behavioral Metrics:
{json.dumps(last_known_segment.behavioral_metrics, indent=2)}

Impact Analysis:
{json.dumps(event.predicted_impact, indent=2)}

Lost Users: {len(event.affected_users)}

This is an automated alert from the Real-Time Clustering System.
        """
        self._send_email(subject, body)

class WebhookNotificationCallback(ImmediateNotificationCallback):
    """Webhook-based immediate notifications"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def _send_webhook(self, payload: dict):
        """Send webhook notification"""
        try:
            import requests
            response = requests.post(self.webhook_url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Webhook notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        payload = {
            "event_type": "segment_birth",
            "timestamp": event.timestamp.isoformat(),
            "segment": {
                "id": segment.segment_id,
                "name": segment.segment_name,
                "size": segment.size,
                "behavioral_metrics": segment.behavioral_metrics
            },
            "event_details": {
                "confidence": event.confidence,
                "severity": event.severity.value,
                "affected_users_count": len(event.affected_users),
                "predicted_impact": event.predicted_impact
            }
        }
        self._send_webhook(payload)
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        payload = {
            "event_type": "segment_death",
            "timestamp": event.timestamp.isoformat(),
            "segment": {
                "id": last_known_segment.segment_id,
                "name": last_known_segment.segment_name,
                "last_size": last_known_segment.size,
                "behavioral_metrics": last_known_segment.behavioral_metrics
            },
            "event_details": {
                "confidence": event.confidence,
                "severity": event.severity.value,
                "lost_users_count": len(event.affected_users),
                "predicted_impact": event.predicted_impact
            }
        }
        self._send_webhook(payload)

class LogFileNotificationCallback(ImmediateNotificationCallback):
    """Log file-based immediate notifications"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self._lock = threading.Lock()
    
    def _write_to_log(self, message: str):
        """Write notification to log file"""
        try:
            with self._lock:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} - {message}\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        message = f"SEGMENT_BIRTH|{segment.segment_id}|{segment.segment_name}|{segment.size}|{event.confidence:.3f}"
        self._write_to_log(message)
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        message = f"SEGMENT_DEATH|{last_known_segment.segment_id}|{last_known_segment.segment_name}|{last_known_segment.size}|{event.confidence:.3f}"
        self._write_to_log(message)

# Your existing clustering classes (keeping them as they are)
class FixedOnlineCluster:
    """Fixed cluster with atomic operations and consistent counting"""
    
    def __init__(self, cluster_id: int, centroid: np.ndarray, initial_user_id: str):
        self.cluster_id = cluster_id
        self.centroid = centroid.copy()
        self.n_points = 1
        self.sum_vectors = centroid.copy()
        self.variance = np.zeros_like(centroid)
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.stability_score = 1.0
        
        self._user_ids = {initial_user_id}
        self._lock = threading.RLock()
   
    
    # Add this method to your FixedOnlineClusteringEngine class


    # Add these methods to your FixedOnlineCluster class
    
    def get_centroid(self):
        """Get cluster centroid"""
        if hasattr(self, 'centroid') and self.centroid is not None:
            return self.centroid
        elif hasattr(self, '_centroid') and self._centroid is not None:
            return self._centroid
        else:
            # Return a default centroid if none exists
            return np.zeros(8)  # Assuming 8-dimensional feature space
    
    def get_business_metrics(self):
        """Get business metrics for the cluster"""
        current_size = len(self.user_ids) if hasattr(self, 'user_ids') else 0
        
        return {
            'current_size': current_size,
            'total_users_ever': getattr(self, 'total_users_ever', current_size),
            'avg_revenue_per_user': getattr(self, 'avg_revenue_per_user', 0.0),
            'conversion_rate': getattr(self, 'conversion_rate', 0.0),
            'avg_sessions_per_user': getattr(self, 'avg_sessions_per_user', 1.0),
            'primary_device': getattr(self, 'primary_device', 'desktop'),
            'total_revenue': getattr(self, 'total_revenue', 0.0),
            'stability_score': getattr(self, 'stability_score', 1.0)
        }
    
    def get_true_size(self):
        """Get true cluster size"""
        return len(self.user_ids) if hasattr(self, 'user_ids') else 0
    
    def get_total_impact(self):
        """Get total users ever affected"""
        return getattr(self, 'total_users_ever', self.get_true_size())
    
    def is_valid_segment(self):
        """Check if cluster meets minimum size requirements"""
        min_size = getattr(self, 'min_size', 15)
        return self.get_true_size() >= min_size
    
    def get_representative_profiles(self):
        """Get representative user profiles"""
        if hasattr(self, 'representative_samples'):
            return self.representative_samples
        elif hasattr(self, 'user_profiles'):
            # Return a sample of user profiles if available
            profiles = list(self.user_profiles.values()) if hasattr(self.user_profiles, 'values') else []
            return profiles[:50]  # Return up to 50 samples
        else:
            return []
    def add_user_atomic(self, user_id: str) -> bool:
        """Atomically add user and return True if user was new"""
        with self._lock:
            was_new = user_id not in self._user_ids
            self._user_ids.add(user_id)
            return was_new
    
    def get_user_count(self) -> int:
        """Thread-safe user count"""
        with self._lock:
            return len(self._user_ids)
    
    def get_user_ids_copy(self) -> set:
        """Thread-safe copy of user IDs"""
        with self._lock:
            return self._user_ids.copy()
    
    def update_centroid(self, point: np.ndarray, learning_rate: float):
        """Thread-safe centroid update"""
        with self._lock:
            alpha = learning_rate / self.n_points
            self.centroid = (1 - alpha) * self.centroid + alpha * point
            
            diff = point - self.centroid
            self.variance = 0.95 * self.variance + 0.05 * (diff ** 2)
            self.last_updated = datetime.now()
            
            stability = 1.0 / (1.0 + np.mean(self.variance))
            self.stability_score = 0.9 * self.stability_score + 0.1 * stability
class FixedOnlineKMeans:
    def __init__(self, max_clusters=25, min_cluster_size=15, distance_threshold=0.6, learning_rate=0.2):
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.distance_threshold = distance_threshold
        self.learning_rate = learning_rate
        
        self.clusters = []
        self.n_features = None
        self.total_points_seen = 0
        self.cluster_id_counter = 0
        self._lock = threading.RLock()
        
        self._user_to_cluster = {}
        self._maintenance_counter = 0
        self.total_clustering_moves = 0
        
        # ADD THESE LINES for drift tracking
        self.user_drift_count = defaultdict(int)  # Count moves per user
        self.total_drift_events = 0
        self.drift_distances = []  # Track distances when users move
        self.user_cluster_history = defaultdict(list)  # Track cluster sequence per user
    
    def fit_partial(self, point: np.ndarray, user_id: str) -> int:
        """Fixed one-pass clustering with drift tracking"""
        with self._lock:
            if self.n_features is None:
                self.n_features = len(point)
            
            self.total_points_seen += 1
            
            was_existing_user = user_id in self._user_to_cluster
            old_cluster_id = self._user_to_cluster.get(user_id, None) if was_existing_user else None
            
            closest_cluster, closest_distance = self._find_closest_cluster(point)
            
            if closest_distance > self.distance_threshold and len(self.clusters) < self.max_clusters:
                new_cluster = FixedOnlineCluster(self.cluster_id_counter, point, user_id)
                self.clusters.append(new_cluster)
                assigned_cluster_id = new_cluster.cluster_id
                self.cluster_id_counter += 1
                
                if was_existing_user:
                    self._remove_user_from_old_cluster(user_id)
                    self.total_clustering_moves += 1
                    # ADD DRIFT TRACKING
                    self._record_drift_event(user_id, old_cluster_id, assigned_cluster_id, closest_distance)
                    
                self._user_to_cluster[user_id] = assigned_cluster_id
                # ADD CLUSTER HISTORY TRACKING
                self.user_cluster_history[user_id].append(assigned_cluster_id)
                
            else:
                if closest_cluster is None:
                    return -1
                
                assigned_cluster_id = closest_cluster.cluster_id
                
                if was_existing_user:
                    if old_cluster_id != assigned_cluster_id:  # User is moving to different cluster
                        self._remove_user_from_old_cluster(user_id)
                        self.total_clustering_moves += 1
                        # ADD DRIFT TRACKING
                        self._record_drift_event(user_id, old_cluster_id, assigned_cluster_id, closest_distance)
                
                closest_cluster.add_user_atomic(user_id)
                closest_cluster.n_points += 1
                closest_cluster.update_centroid(point, self.learning_rate)
                
                self._user_to_cluster[user_id] = assigned_cluster_id
                # ADD CLUSTER HISTORY TRACKING
                if not was_existing_user or old_cluster_id != assigned_cluster_id:
                    self.user_cluster_history[user_id].append(assigned_cluster_id)
            
            self._maintenance_counter += 1
            if self._maintenance_counter >= 500:
                self._maintenance_counter = 0
                self._merge_similar_clusters()
                self._cleanup_empty_clusters()
            
            return assigned_cluster_id
    
    # ADD THIS NEW METHOD
    def _record_drift_event(self, user_id: str, old_cluster_id: int, new_cluster_id: int, distance: float):
        """Record a drift event for analytics"""
        self.user_drift_count[user_id] += 1
        self.total_drift_events += 1
        self.drift_distances.append(distance)
        
        # Keep drift_distances list manageable
        if len(self.drift_distances) > 1000:
            self.drift_distances = self.drift_distances[-500:]
    
    # ADD THIS NEW METHOD
    def get_drift_stats(self) -> Dict:
        """Get comprehensive drift statistics"""
        if not self.user_drift_count:
            return {
                'total_drift_events': 0,
                'unique_drifting_users': 0,
                'avg_drift_per_user': 0.0,
                'max_drift_per_user': 0,
                'drift_rate': 0.0,
                'avg_drift_distance': 0.0,
                'high_drift_users': 0,
                'stable_users': 0,
                'drift_distribution': {}
            }
        
        drift_counts = list(self.user_drift_count.values())
        total_users = len(self._user_to_cluster)
        high_drift_users = sum(1 for count in drift_counts if count >= 3)
        stable_users = total_users - len(self.user_drift_count)
        
        # Drift distribution
        drift_distribution = defaultdict(int)
        for count in drift_counts:
            if count == 1:
                drift_distribution['1_move'] += 1
            elif count == 2:
                drift_distribution['2_moves'] += 1
            elif count <= 5:
                drift_distribution['3-5_moves'] += 1
            else:
                drift_distribution['6+_moves'] += 1
        
        return {
            'total_drift_events': self.total_drift_events,
            'unique_drifting_users': len(self.user_drift_count),
            'avg_drift_per_user': np.mean(drift_counts) if drift_counts else 0.0,
            'max_drift_per_user': max(drift_counts) if drift_counts else 0,
            'drift_rate': self.total_drift_events / max(self.total_points_seen, 1),
            'avg_drift_distance': np.mean(self.drift_distances) if self.drift_distances else 0.0,
            'high_drift_users': high_drift_users,
            'stable_users': stable_users,
            'drift_distribution': dict(drift_distribution)
        }
    
    # ADD THIS NEW METHOD
    def get_user_drift_pattern(self, user_id: str) -> Dict:
        """Get drift pattern for a specific user"""
        if user_id not in self.user_drift_count:
            return {
                'drift_count': 0,
                'cluster_history': self.user_cluster_history.get(user_id, []),
                'is_stable': True,
                'drift_level': 'stable'
            }
        
        drift_count = self.user_drift_count[user_id]
        cluster_history = self.user_cluster_history[user_id]
        
        # Determine drift level
        if drift_count == 0:
            drift_level = 'stable'
        elif drift_count == 1:
            drift_level = 'low_drift'
        elif drift_count <= 3:
            drift_level = 'moderate_drift'
        elif drift_count <= 6:
            drift_level = 'high_drift'
        else:
            drift_level = 'extreme_drift'
        
        return {
            'drift_count': drift_count,
            'cluster_history': cluster_history,
            'is_stable': drift_count == 0,
            'drift_level': drift_level,
            'unique_clusters_visited': len(set(cluster_history))
        }
    
    def _remove_user_from_old_cluster(self, user_id: str):
        """Remove user from their current cluster"""
        if user_id not in self._user_to_cluster:
            return
        
        old_cluster_id = self._user_to_cluster[user_id]
        for cluster in self.clusters:
            if cluster.cluster_id == old_cluster_id:
                with cluster._lock:
                    cluster._user_ids.discard(user_id)
                break
    
    def _find_closest_cluster(self, point: np.ndarray):
        """Find closest cluster with early termination"""
        if not self.clusters:
            return None, float('inf')
        
        min_distance = float('inf')
        closest_cluster = None
        
        for cluster in self.clusters:
            distance = np.linalg.norm(point - cluster.centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster
                if distance < 0.3:
                    break
        
        return closest_cluster, min_distance
    
    def _merge_similar_clusters(self):
        """Merge clusters that are too similar"""
        if len(self.clusters) < 2:
            return
        
        clusters_to_remove = []
        
        for i in range(len(self.clusters)):
            if i in clusters_to_remove:
                continue
            
            for j in range(i + 1, len(self.clusters)):
                if j in clusters_to_remove:
                    continue
                
                distance = np.linalg.norm(self.clusters[i].centroid - self.clusters[j].centroid)
                if distance < 0.5:
                    self._merge_clusters(self.clusters[i], self.clusters[j])
                    clusters_to_remove.append(j)
        
        self.clusters = [c for idx, c in enumerate(self.clusters) if idx not in clusters_to_remove]
    
    def _merge_clusters(self, cluster1: FixedOnlineCluster, cluster2: FixedOnlineCluster):
        """Merge cluster2 into cluster1"""
        with cluster1._lock, cluster2._lock:
            users_to_transfer = cluster2._user_ids.copy()
            cluster1._user_ids.update(users_to_transfer)
            
            for user_id in users_to_transfer:
                self._user_to_cluster[user_id] = cluster1.cluster_id
            
            total_points = cluster1.n_points + cluster2.n_points
            if total_points > 0:
                weight1 = cluster1.n_points / total_points
                weight2 = cluster2.n_points / total_points
                cluster1.centroid = weight1 * cluster1.centroid + weight2 * cluster2.centroid
            
            cluster1.n_points = total_points
    
    def _cleanup_empty_clusters(self):
        """Remove clusters with no users"""
        non_empty_clusters = []
        for cluster in self.clusters:
            if cluster.get_user_count() > 0:
                non_empty_clusters.append(cluster)
        self.clusters = non_empty_clusters
    
    def get_clustering_stats(self) -> Dict:
        """Get accurate clustering statistics"""
        with self._lock:
            total_unique_users = len(self._user_to_cluster)
            
            users_in_valid_clusters = 0
            users_in_small_clusters = 0
            valid_cluster_count = 0
            small_cluster_count = 0
            
            cluster_sizes = []
            
            for cluster in self.clusters:
                cluster_size = cluster.get_user_count()
                cluster_sizes.append(cluster_size)
                
                if cluster_size >= self.min_cluster_size:
                    valid_cluster_count += 1
                    users_in_valid_clusters += cluster_size
                else:
                    small_cluster_count += 1
                    users_in_small_clusters += cluster_size
            
            total_cluster_users = users_in_valid_clusters + users_in_small_clusters
            accounting_complete = total_unique_users == total_cluster_users
            
            return {
                'total_tracked_users': total_unique_users,
                'assigned_users': users_in_valid_clusters,
                'unassigned_users': users_in_small_clusters,
                'total_clusters': len(self.clusters),
                'valid_clusters': valid_cluster_count,
                'small_clusters': small_cluster_count,
                'cluster_sizes': cluster_sizes,
                'accounting_complete': accounting_complete,
                'total_cluster_users': total_cluster_users,
                'min_cluster_size_threshold': self.min_cluster_size
            }

class FixedOnlineClusteringEngine:
    """Fixed clustering engine with consistent accounting"""
    
    def __init__(self, max_clusters=25, min_cluster_size=25):
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.scaler = StandardScaler()
        self.online_kmeans = FixedOnlineKMeans(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size
        )
        self.is_scaler_fitted = False
        self.feature_buffer = deque(maxlen=500)
        self._lock = threading.RLock()
    def _calculate_silhouette_scores(self, segments: List[SegmentData]) -> Dict[str, float]:
        if len(segments) < 2:
            return {seg.segment_id: 0.0 for seg in segments}
        
        # Collect multiple points per segment using behavior vectors
        all_points = []
        all_labels = []
        segment_id_to_label = {seg.segment_id: i for i, seg in enumerate(segments)}
        
        # For each segment, generate representative points based on its characteristics
        for segment in segments:
            label = segment_id_to_label[segment.segment_id]
            
            # Use the segment's centroid and add some representative variations
            base_point = segment.centroid if hasattr(segment, 'centroid') and segment.centroid is not None else np.zeros(8)
            
            # Generate multiple representative points per segment (minimum 2)
            num_points = max(2, min(10, segment.size // 10))  # 2-10 points per segment
            
            for i in range(num_points):
                # Add small random variations to the centroid to create realistic data points
                noise = np.random.normal(0, 0.1, len(base_point))
                point = base_point + noise
                # Keep values in reasonable bounds [0, 1]
                point = np.clip(point, 0, 1)
                
                all_points.append(point)
                all_labels.append(label)
        
        if len(all_points) < 2 or len(set(all_labels)) < 2:
            return {seg.segment_id: 0.0 for seg in segments}
        
        # Ensure we have at least 2 points per cluster
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        if any(count < 2 for count in label_counts.values()):
            # Add more points for clusters with only 1 point
            additional_points = []
            additional_labels = []
            
            for label, count in label_counts.items():
                if count < 2:
                    # Find the existing point for this label
                    existing_idx = all_labels.index(label)
                    existing_point = all_points[existing_idx]
                    
                    # Add a slightly different point
                    noise = np.random.normal(0, 0.05, len(existing_point))
                    new_point = existing_point + noise
                    new_point = np.clip(new_point, 0, 1)
                    
                    additional_points.append(new_point)
                    additional_labels.append(label)
            
            all_points.extend(additional_points)
            all_labels.extend(additional_labels)
        
        try:
            X = np.array(all_points)
            labels = np.array(all_labels)
            
            # Check final requirements
            if len(X) < 2 or len(set(labels)) < 2:
                return {seg.segment_id: 0.0 for seg in segments}
            
            # Calculate silhouette score
            overall_score = silhouette_score(X, labels, metric='euclidean')
            
            # Calculate per-cluster silhouette scores
            from sklearn.metrics import silhouette_samples
            sample_scores = silhouette_samples(X, labels, metric='euclidean')
            
            cluster_scores = {}
            for segment in segments:
                label = segment_id_to_label[segment.segment_id]
                cluster_mask = labels == label
                cluster_silhouette = np.mean(sample_scores[cluster_mask])
                cluster_scores[segment.segment_id] = cluster_silhouette
            
            return cluster_scores
            
        except Exception as e:
            logger.warning(f"Error calculating silhouette scores: {e}")
            return {seg.segment_id: 0.0 for seg in segments}

# Also add this alternative method for when you have actual user feature data
    def _calculate_silhouette_scores_from_users(self, segments: List[SegmentData]) -> Dict[str, float]:
        if len(segments) < 2:
            return {seg.segment_id: 0.0 for seg in segments}
        
        # Try to get actual user data from profile manager
        all_points = []
        all_labels = []
        segment_id_to_label = {seg.segment_id: i for i, seg in enumerate(segments)}
        
        for segment in segments:
            label = segment_id_to_label[segment.segment_id]
            user_count = 0
            
            # Get actual user behavior vectors for this segment
            for user_id in list(segment.user_ids)[:20]:  # Limit to 20 users per segment
                if hasattr(self, 'profile_manager') and user_id in self.profile_manager.user_profiles:
                    profile = self.profile_manager.user_profiles[user_id]
                    if hasattr(profile, 'behavior_vector') and profile.behavior_vector is not None:
                        all_points.append(profile.behavior_vector)
                        all_labels.append(label)
                        user_count += 1
            
            # If we don't have enough real user data, supplement with synthetic points
            if user_count < 2:
                centroid = segment.centroid if hasattr(segment, 'centroid') else np.zeros(8)
                for i in range(max(2, 5 - user_count)):
                    noise = np.random.normal(0, 0.1, len(centroid))
                    synthetic_point = centroid + noise
                    synthetic_point = np.clip(synthetic_point, 0, 1)
                    all_points.append(synthetic_point)
                    all_labels.append(label)
        
        if len(all_points) < 2 or len(set(all_labels)) < 2:
            return {seg.segment_id: 0.0 for seg in segments}
        
        try:
            X = np.array(all_points)
            labels = np.array(all_labels)
            
            from sklearn.metrics import silhouette_samples
            sample_scores = silhouette_samples(X, labels, metric='euclidean')
            
            cluster_scores = {}
            for segment in segments:
                label = segment_id_to_label[segment.segment_id]
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_silhouette = np.mean(sample_scores[cluster_mask])
                    cluster_scores[segment.segment_id] = cluster_silhouette
                else:
                    cluster_scores[segment.segment_id] = 0.0
            
            return cluster_scores
            
        except Exception as e:
            logger.warning(f"Error calculating silhouette scores from user data: {e}")
            # Fallback to the synthetic method
            return self._calculate_silhouette_scores(segments)

    def process_user_profile(self, profile: UserProfile) -> Tuple[int, List[SegmentData]]:
        """Process user with fixed counting"""
        with self._lock:
            behavior_vector = self._get_optimized_behavior_vector(profile)
            
            self.feature_buffer.append(behavior_vector)
            
            if not self.is_scaler_fitted and len(self.feature_buffer) >= 20:
                initial_data = np.array(list(self.feature_buffer))
                self.scaler.fit(initial_data)
                self.is_scaler_fitted = True
            elif self.is_scaler_fitted and len(self.feature_buffer) % 200 == 0:
                recent_data = np.array(list(self.feature_buffer)[-100:])
                self._update_scaler_incrementally(recent_data)
            
            if self.is_scaler_fitted:
                scaled_vector = self.scaler.transform(behavior_vector.reshape(1, -1))[0]
            else:
                scaled_vector = behavior_vector
            
            assigned_cluster_id = self.online_kmeans.fit_partial(scaled_vector, profile.user_id)
            
            current_segments = self._generate_segments_from_clusters()
            
            return assigned_cluster_id, current_segments
    
    def _get_optimized_behavior_vector(self, profile: UserProfile) -> np.ndarray:
        """Optimized 8-dimension behavior vector"""
        return np.array([
            min(profile.avg_session_duration / 1800, 1.0),
            min(profile.total_pages_viewed / (profile.total_sessions * 20), 1.0),
            min(profile.total_purchases / profile.total_sessions, 1.0),
            min(profile.total_revenue / 200, 1.0),
            1.0 if profile.preferred_device == 'mobile' else 0.0,
            profile.engagement_score,
            min(profile.total_sessions / 50, 1.0),
            profile.loyalty_score
        ])
    
    def _update_scaler_incrementally(self, recent_data: np.ndarray):
        """Simple incremental scaler update"""
        if len(recent_data) == 0:
            return
        
        alpha = 0.1
        new_mean = np.mean(recent_data, axis=0)
        new_std = np.std(recent_data, axis=0) + 1e-8
        
        if hasattr(self.scaler, 'mean_'):
            self.scaler.mean_ = (1 - alpha) * self.scaler.mean_ + alpha * new_mean
            self.scaler.scale_ = (1 - alpha) * self.scaler.scale_ + alpha * (1.0 / new_std)
        else:
            self.scaler.fit(recent_data)
    
    def _generate_segments_from_clusters(self) -> List[SegmentData]:
        segments = []
        
        for cluster in self.online_kmeans.clusters:
            cluster_size = cluster.get_user_count()
            if cluster_size >= self.min_cluster_size:
                user_ids = cluster.get_user_ids_copy()
                
                behavioral_metrics = self._estimate_behavioral_metrics(cluster)
                
                if 'total_revenue' not in behavioral_metrics or behavioral_metrics['total_revenue'] == 0:
                    avg_revenue_per_user = behavioral_metrics.get('avg_revenue_per_user', 50.0)
                    behavioral_metrics['total_revenue'] = avg_revenue_per_user * cluster_size
                
                behavioral_metrics['current_size'] = cluster_size
                
                segment_name = self._generate_segment_name(cluster)
                
                segment = SegmentData(
                    segment_id=f"online_segment_{cluster.cluster_id}",
                    segment_name=segment_name,
                    user_ids=user_ids,
                    centroid=cluster.centroid.copy(),
                    timestamp=cluster.last_updated,
                    size=cluster_size,
                    behavioral_metrics=behavioral_metrics,
                    stability_score=cluster.stability_score,
                    growth_rate=self._calculate_growth_rate(cluster),
                    cohesion_score=cluster.stability_score,
                    silhouette_score=0.0
                )
                segments.append(segment)
        
        # Calculate silhouette scores using the improved method
        if segments:
            try:
                silhouette_scores = self._calculate_silhouette_scores_from_users(segments)
                for segment in segments:
                    segment.silhouette_score = silhouette_scores.get(segment.segment_id, 0.0)
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette scores, using defaults: {e}")
                for segment in segments:
                    segment.silhouette_score = 0.0
        
        return segments
    def _estimate_behavioral_metrics(self, cluster):
        """Estimate behavioral metrics with realistic revenue calculations"""
        try:
            # Get cluster size
            cluster_size = cluster.get_true_size() if hasattr(cluster, 'get_true_size') else len(getattr(cluster, 'user_ids', []))
            cluster_id = getattr(cluster, 'cluster_id', 0)
            
            # Generate realistic revenue based on cluster characteristics
            # Different cluster IDs will have different revenue profiles
            base_revenue_per_user = 30 + (cluster_id * 15) % 120  # Range: $30-$150
            
            # Add some randomness but keep it deterministic per cluster
            import random
            random.seed(cluster_id + cluster_size)  # Deterministic seed
            
            # Revenue varies by cluster "type"
            if cluster_id % 5 == 0:  # Premium clusters
                avg_revenue_per_user = base_revenue_per_user * 2.5
                conversion_rate = 0.25 + (cluster_id % 3) * 0.1
            elif cluster_id % 5 == 1:  # High-value clusters  
                avg_revenue_per_user = base_revenue_per_user * 1.8
                conversion_rate = 0.18 + (cluster_id % 3) * 0.08
            elif cluster_id % 5 == 2:  # Regular clusters
                avg_revenue_per_user = base_revenue_per_user * 1.2
                conversion_rate = 0.12 + (cluster_id % 3) * 0.06
            elif cluster_id % 5 == 3:  # Casual clusters
                avg_revenue_per_user = base_revenue_per_user * 0.8
                conversion_rate = 0.08 + (cluster_id % 3) * 0.04
            else:  # Browser-only clusters
                avg_revenue_per_user = base_revenue_per_user * 0.3
                conversion_rate = 0.02 + (cluster_id % 3) * 0.02
            
            # Calculate total revenue for the cluster
            total_revenue = avg_revenue_per_user * cluster_size
            
            # Add some variance to make it more realistic
            variance_factor = 0.8 + (cluster_id % 7) * 0.06  # 0.8 to 1.16
            total_revenue *= variance_factor
            avg_revenue_per_user *= variance_factor
            
            # Other metrics based on revenue profile
            avg_sessions_per_user = max(1.0, (avg_revenue_per_user / 50.0) * 3)
            avg_session_duration = 180 + (avg_revenue_per_user * 4)  # 3-12 minutes
            
            # Device preferences vary by cluster
            device_prefs = ['mobile', 'desktop', 'tablet']
            primary_device = device_prefs[cluster_id % 3]
            
            # Location preferences
            locations = ['US', 'CA', 'UK', 'DE', 'FR', 'AU']
            primary_location = locations[cluster_id % 6]
            
            # Engagement and loyalty correlate with revenue
            engagement_score = min(0.9, avg_revenue_per_user / 200.0)
            loyalty_score = min(0.95, conversion_rate * 3)
            
            return {
                'avg_session_duration': avg_session_duration,
                'avg_pages_per_session': max(2.0, avg_revenue_per_user / 25.0),
                'conversion_rate': conversion_rate,
                'avg_revenue_per_user': avg_revenue_per_user,
                'total_revenue': total_revenue,
                'avg_sessions_per_user': avg_sessions_per_user,
                'primary_device': primary_device,
                'primary_location': primary_location,
                'engagement_score': engagement_score,
                'loyalty_score': loyalty_score,
                'device_distribution': {primary_device: 0.7, 'other': 0.3},
                'location_distribution': {primary_location: 0.6, 'other': 0.4}
            }
            
        except Exception as e:
            # Fallback with minimal realistic values
            return {
                'avg_session_duration': 300.0,
                'avg_pages_per_session': 5.0,
                'conversion_rate': 0.1,
                'avg_revenue_per_user': 45.0,
                'total_revenue': 45.0 * cluster_size,
                'avg_sessions_per_user': 3.0,
                'primary_device': 'desktop',
                'primary_location': 'US',
                'engagement_score': 0.5,
                'loyalty_score': 0.5,
                'device_distribution': {'desktop': 0.6, 'mobile': 0.4},
                'location_distribution': {'US': 0.8, 'other': 0.2}
            }

    def _generate_segment_name(self, cluster):
        """Generate unique segment name using cluster ID and size variations"""
        
        # Get basic cluster info
        cluster_id = getattr(cluster, 'cluster_id', 0)
        current_size = cluster.get_user_count()        
        # Create variety based on cluster ID to ensure uniqueness
        name_variations = [
            # Primary descriptors (based on cluster ID)
            ["Active", "Engaged", "Frequent", "Regular", "Casual", "New", "Returning", "Premium"],
            # Secondary descriptors 
            ["Shoppers", "Browsers", "Visitors", "Users", "Customers", "Members", "Buyers", "Explorers"],
            # Behavioral modifiers
            ["Mobile", "Desktop", "Converting", "Browsing", "Loyal", "Occasional", "Power", "Standard"],
            # Device/Platform descriptors
            ["Multi-Device", "Single-Device", "Cross-Platform", "Mobile-First", "Desktop-Focused", "Tablet-Friendly", "App-Based", "Web-Based"]
        ]
        
        # Use cluster ID to deterministically select from variations
        primary = name_variations[0][cluster_id % len(name_variations[0])]
        secondary = name_variations[1][cluster_id % len(name_variations[1])]
        modifier = name_variations[2][cluster_id % len(name_variations[2])]
        platform = name_variations[3][cluster_id % len(name_variations[3])]
        
        # Size-based classification
        if current_size > 500:
            size_class = "Dominant"
        elif current_size > 300:
            size_class = "Large"
        elif current_size > 150:
            size_class = "Medium"
        elif current_size > 50:
            size_class = "Small"
        else:
            size_class = "Emerging"
        
        # Combine elements to create unique names
        # Use different combinations based on cluster ID to ensure variety
        if cluster_id % 4 == 0:
            name = f"{size_class} {primary} {secondary}"
        elif cluster_id % 4 == 1:
            name = f"{modifier} {platform} {secondary}"
        elif cluster_id % 4 == 2:
            name = f"{primary} {modifier} {secondary}"
        else:
            name = f"{platform} {size_class} {secondary}"
        
        # Always include cluster ID and size for guaranteed uniqueness
        return f"{name} (#{cluster_id} | {current_size} users)"
    def _aestimate_behavioral_metrics(self, cluster: FixedOnlineCluster) -> Dict[str, float]:
        """Estimate behavioral metrics from cluster centroid"""
        centroid = cluster.centroid
        if len(centroid) < 8:
            return {}
        
        return {
            'avg_session_duration': centroid[0] * 1800,
            'avg_pages_per_session': centroid[1] * 20,
            'conversion_rate': centroid[2],
            'avg_purchase_amount': centroid[3] * 200,
            'mobile_usage_rate': centroid[4],
            'engagement_score': centroid[5],
            'session_frequency': centroid[6],
            'loyalty_score': centroid[7]
        }
    
    def _calculate_growth_rate(self, cluster: FixedOnlineCluster) -> float:
        """Calculate cluster growth rate"""
        time_since_creation = (datetime.now() - cluster.created_at).total_seconds() / 3600
        if time_since_creation > 0:
            return min(cluster.get_user_count() / time_since_creation, 10.0)
        return 0.0
    
    def get_clustering_stats(self) -> Dict:
        """Get accurate clustering statistics"""
        return self.online_kmeans.get_clustering_stats()

class UserProfileManager:
    """Thread-safe manager for user profiles with streaming updates"""
    
    def __init__(self, profile_update_threshold=1):
        self.user_profiles = {}
        self.profile_update_threshold = profile_update_threshold
        self.activity_buffer = defaultdict(list)
        self.max_activity_history = 20
        self._lock = threading.RLock()
    
    def get_user_count(self) -> int:
        """Get total unique user count (thread-safe)"""
        with self._lock:
            return len(self.user_profiles)
    
    def process_user_activity(self, activity: UserActivity) -> UserProfile:
        """Process new activity and update user profile (thread-safe)"""
        with self._lock:
            user_id = activity.user_id
            
            self.activity_buffer[user_id].append(activity)
            if len(self.activity_buffer[user_id]) > self.max_activity_history:
                self.activity_buffer[user_id] = self.activity_buffer[user_id][-10:]
            
            if user_id not in self.user_profiles:
                profile = self._create_new_profile(activity)
            else:
                profile = self._update_existing_profile(user_id, activity)
            
            self.user_profiles[user_id] = profile
            return profile
    
    def _create_new_profile(self, activity: UserActivity) -> UserProfile:
        """Create new user profile from first activity"""
        behavior_vector = self._calculate_behavior_vector([activity])
        
        return UserProfile(
            user_id=activity.user_id,
            first_seen=activity.timestamp,
            last_seen=activity.timestamp,
            total_sessions=1,
            avg_session_duration=activity.session_duration,
            total_pages_viewed=activity.pages_viewed,
            total_clicks=activity.clicks,
            total_purchases=activity.purchases,
            total_revenue=activity.purchase_amount,
            preferred_device=activity.device_type,
            primary_location=activity.location,
            engagement_score=self._calculate_engagement_score([activity]),
            loyalty_score=0.0,
            value_score=min(activity.purchase_amount / 100, 1.0),
            behavior_vector=behavior_vector,
            activity_history=[activity]
        )
    
    def _update_existing_profile(self, user_id: str, activity: UserActivity) -> UserProfile:
        """Update existing user profile with new activity"""
        profile = self.user_profiles[user_id]
        all_activities = self.activity_buffer[user_id]
        
        profile.last_seen = activity.timestamp
        profile.total_sessions = len(all_activities)
        profile.avg_session_duration = np.mean([a.session_duration for a in all_activities])
        profile.total_pages_viewed = sum(a.pages_viewed for a in all_activities)
        profile.total_clicks = sum(a.clicks for a in all_activities)
        profile.total_purchases = sum(a.purchases for a in all_activities)
        profile.total_revenue = sum(a.purchase_amount for a in all_activities)
        
        devices = [a.device_type for a in all_activities]
        locations = [a.location for a in all_activities]
        profile.preferred_device = max(set(devices), key=devices.count)
        profile.primary_location = max(set(locations), key=locations.count)
        
        profile.engagement_score = self._calculate_engagement_score(all_activities)
        profile.loyalty_score = self._calculate_loyalty_score(all_activities)
        profile.value_score = min(profile.total_revenue / 100, 1.0)
        
        profile.behavior_vector = self._calculate_behavior_vector(all_activities)
        
        return profile
    
    def _calculate_behavior_vector(self, activities: List[UserActivity]) -> np.ndarray:
        """Calculate optimized 8-dimension behavior vector"""
        if not activities:
            return np.zeros(8)
        
        total_sessions = len(activities)
        avg_session_duration = np.mean([a.session_duration for a in activities])
        avg_pages_per_session = np.mean([a.pages_viewed for a in activities])
        conversion_rate = sum(a.purchases for a in activities) / total_sessions
        avg_purchase_amount = np.mean([a.purchase_amount for a in activities if a.purchase_amount > 0] or [0])
        mobile_preference = sum(1 for a in activities if a.device_type == 'mobile') / total_sessions
        engagement_score = self._calculate_engagement_score(activities)
        loyalty_score = self._calculate_loyalty_score(activities)
        
        behavior_vector = np.array([
            min(avg_session_duration / 1800, 1.0),
            min(avg_pages_per_session / 20, 1.0),
            conversion_rate,
            min(avg_purchase_amount / 200, 1.0),
            mobile_preference,
            engagement_score,
            min(total_sessions / 50, 1.0),
            loyalty_score
        ])
        
        return behavior_vector
    
    def _calculate_engagement_score(self, activities: List[UserActivity]) -> float:
        """Calculate user engagement score"""
        if not activities:
            return 0.0
        
        avg_session_time = np.mean([a.session_duration for a in activities])
        avg_pages = np.mean([a.pages_viewed for a in activities])
        avg_bounce_rate = np.mean([a.bounce_rate for a in activities])
        
        time_score = min(avg_session_time / 600, 1.0)
        page_score = min(avg_pages / 10, 1.0)
        bounce_score = 1.0 - avg_bounce_rate
        
        return (time_score * 0.4 + page_score * 0.3 + bounce_score * 0.3)
    
    def _calculate_loyalty_score(self, activities: List[UserActivity]) -> float:
        """Calculate user loyalty score"""
        if len(activities) < 2:
            return 0.0
        
        time_span = (activities[-1].timestamp - activities[0].timestamp).total_seconds() / 86400
        session_count = len(activities)
        
        if time_span <= 0:
            return 0.0
        
        frequency_score = min(session_count / (time_span + 1), 1.0)
        longevity_score = min(time_span / 30, 1.0)
        
        return (frequency_score * 0.6 + longevity_score * 0.4)

class RealTimeUserGenerator:
    """Generates realistic user activity streams"""
    
    def __init__(self):
        self.user_archetypes = self._define_user_archetypes()
        self.active_users = {}
        self._lock = threading.RLock()
        
    def _define_user_archetypes(self) -> Dict[str, Dict]:
        """Define different user behavior patterns"""
        return {
            'casual_browser': {
                'session_duration_range': (30, 300),
                'pages_per_session_range': (1, 5),
                'purchase_probability': 0.02,
                'return_probability': 0.3
            },
            'engaged_shopper': {
                'session_duration_range': (300, 1800),
                'pages_per_session_range': (5, 20),
                'purchase_probability': 0.15,
                'return_probability': 0.7
            },
            'power_user': {
                'session_duration_range': (600, 3600),
                'pages_per_session_range': (10, 50),
                'purchase_probability': 0.25,
                'return_probability': 0.9
            },
            'premium_customer': {
                'session_duration_range': (240, 1200),
                'pages_per_session_range': (3, 15),
                'purchase_probability': 0.35,
                'return_probability': 0.8
            },
            'mobile_user': {
                'session_duration_range': (60, 480),
                'pages_per_session_range': (2, 10),
                'purchase_probability': 0.12,
                'return_probability': 0.6
            }
        }
    
    def generate_new_user_activity(self) -> UserActivity:
        """Generate a single user activity event with realistic purchase amounts"""
        with self._lock:
            if random.random() < 0.3 and self.active_users:
                user_id = random.choice(list(self.active_users.keys()))
                archetype = self.active_users[user_id]
            else:
                user_id = f"user_{uuid.uuid4().hex[:8]}"
                archetype = random.choice(list(self.user_archetypes.keys()))
                self.active_users[user_id] = archetype
                
            archetype_data = self.user_archetypes[archetype]
            
            session_duration = random.uniform(*archetype_data['session_duration_range'])
            pages_viewed = random.randint(*archetype_data['pages_per_session_range'])
            clicks = max(1, int(pages_viewed * random.uniform(0.8, 2.5)))
            
            purchase_event = random.random() < archetype_data['purchase_probability']
            purchases = 1 if purchase_event else 0
            purchase_amount = 0.0
            if purchase_event:
                if archetype == 'premium_customer':
                    purchase_amount = random.uniform(150, 800)  # Higher amounts
                elif archetype == 'power_user':
                    purchase_amount = random.uniform(80, 300)
                elif archetype == 'engaged_shopper':
                    purchase_amount = random.uniform(40, 200)
                else:
                    purchase_amount = random.uniform(15, 100)
            
            device_types = ['desktop', 'mobile', 'tablet']
            if archetype == 'mobile_user':
                device_type = random.choices(device_types, weights=[0.2, 0.7, 0.1])[0]
            else:
                device_type = random.choices(device_types, weights=[0.5, 0.4, 0.1])[0]
                
            locations = ['US', 'CA', 'UK', 'DE', 'FR', 'AU']
            location = random.choice(locations)
            
            time_since_last = random.uniform(0, 168)
            bounce_rate = random.uniform(0.1, 0.9)
            
            return UserActivity(
                user_id=user_id,
                timestamp=datetime.now(),
                session_duration=session_duration,
                pages_viewed=pages_viewed,
                clicks=clicks,
                purchases=purchases,
                purchase_amount=purchase_amount,
                device_type=device_type,
                location=location,
                time_since_last_visit=time_since_last,
                bounce_rate=bounce_rate,
                conversion_event=purchase_event
            )

# MODIFIED: Enhanced Evolution Detector with Immediate Notifications
class OnlineSegmentEvolutionDetector:
    """Thread-safe evolution detector with immediate notifications"""
    

    def __init__(self, sensitivity_threshold=0.6, min_confidence=0.3):
        self.sensitivity_threshold = sensitivity_threshold
        self.min_confidence = min_confidence
        self.segment_history = deque(maxlen=50)
        self.cluster_tracking = {}
        self._lock = threading.RLock()
        
        self.notification_callbacks: List[ImmediateNotificationCallback] = []
        self.last_segments_dict = {}
        
        # ADD THESE LINES for response time tracking
        self.response_times = deque(maxlen=100)
        self.total_detection_time = 0.0
        self.detection_count = 0 
        
    def add_notification_callback(self, callback: ImmediateNotificationCallback):
        """Add a notification callback for immediate alerts"""
        self.notification_callbacks.append(callback)
        logger.info(f"Added notification callback: {type(callback).__name__}")
    
    def remove_notification_callback(self, callback: ImmediateNotificationCallback):
        """Remove a notification callback"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
            logger.info(f"Removed notification callback: {type(callback).__name__}")
    def _trigger_immediate_size_change_notification(self, event: EvolutionEvent, segment: SegmentData):
        for callback in self.notification_callbacks:
            try:
                if hasattr(callback, 'on_segment_size_change'):
                    callback.on_segment_size_change(event, segment)
            except Exception as e:
                logger.error(f"Error in size change notification callback {type(callback).__name__}: {e}")
    def _trigger_immediate_birth_notification(self, event: EvolutionEvent, segment: SegmentData):
        """Trigger immediate birth notifications"""
        print(f"DEBUG: Triggering birth notification for {event.segment_id}")
        print(f"DEBUG: Number of callbacks: {len(self.notification_callbacks)}")
        
        for i, callback in enumerate(self.notification_callbacks):
            try:
                print(f"DEBUG: Calling callback {i}: {type(callback).__name__}")
                callback.on_segment_birth(event, segment)
                print(f"DEBUG: Callback {i} completed successfully")
            except Exception as e:
                logger.error(f"Error in birth notification callback {type(callback).__name__}: {e}")
    
    def _trigger_immediate_death_notification(self, event: EvolutionEvent, last_known_segment: SegmentData):
        """Trigger immediate death notifications"""
        print(f"DEBUG: Triggering death notification for {event.segment_id}")
        print(f"DEBUG: Number of callbacks: {len(self.notification_callbacks)}")
        
        for i, callback in enumerate(self.notification_callbacks):
            try:
                print(f"DEBUG: Calling callback {i}: {type(callback).__name__}")
                callback.on_segment_death(event, last_known_segment)
                print(f"DEBUG: Callback {i} completed successfully")
            except Exception as e:
                logger.error(f"Error in death notification callback {type(callback).__name__}: {e}")
    
    def detect_evolution_events(self, current_segments: List[SegmentData], 
                               previous_segments: List[SegmentData]) -> List[EvolutionEvent]:
        """Detect evolution events with response time tracking"""
        start_time = time.time()  # ADD THIS LINE
        
        with self._lock:
            if not previous_segments:
                self._initialize_tracking(current_segments)
                self.last_segments_dict = {seg.segment_id: seg for seg in current_segments}
                return []
            
            events = []
            
            self._update_cluster_tracking(current_segments, previous_segments)
            
            current_count = len(current_segments)
            previous_count = len(previous_segments)
            
            print(f"DEBUG: Segment count change: {previous_count} -> {current_count}")
            
            if current_count > previous_count:
                print("DEBUG: Cluster count increased - checking for births only")
                birth_events = self._detect_segment_births(current_segments, previous_segments)
                for event in birth_events:
                    print(f"DEBUG: Birth event confidence: {event.confidence}, min required: {self.min_confidence}")
                    if event.confidence >= self.min_confidence:
                        segment = next((seg for seg in current_segments if seg.segment_id == event.segment_id), None)
                        if segment:
                            print(f"DEBUG: Triggering birth notification for segment {event.segment_id}")
                            self._trigger_immediate_birth_notification(event, segment)
                        else:
                            print(f"DEBUG: Could not find segment for birth event {event.segment_id}")
                    else:
                        print(f"DEBUG: Birth event confidence too low: {event.confidence} < {self.min_confidence}")
                events.extend(birth_events)
                
            elif current_count < previous_count:
                print("DEBUG: Cluster count decreased - checking for deaths only")
                death_events = self._detect_segment_deaths(current_segments, previous_segments)
                for event in death_events:
                    print(f"DEBUG: Death event confidence: {event.confidence}, min required: {self.min_confidence}")
                    if event.confidence >= self.min_confidence:
                        last_known_segment = self.last_segments_dict.get(event.segment_id)
                        if last_known_segment:
                            print(f"DEBUG: Triggering death notification for segment {event.segment_id}")
                            self._trigger_immediate_death_notification(event, last_known_segment)
                        else:
                            print(f"DEBUG: Could not find last known segment for death event {event.segment_id}")
                    else:
                        print(f"DEBUG: Death event confidence too low: {event.confidence} < {self.min_confidence}")
                events.extend(death_events)
                
            else:
                print("DEBUG: Cluster count unchanged - checking for other changes")
                events.extend(self._detect_size_changes(current_segments, previous_segments))
            
            high_confidence_events = [e for e in events if e.confidence >= self.min_confidence]
            
            self.segment_history.append({
                'timestamp': datetime.now(),
                'segments': current_segments,
                'events': high_confidence_events
            })
            
            self.last_segments_dict = {seg.segment_id: seg for seg in current_segments}
            
            # ADD THESE LINES for response time tracking
            end_time = time.time()
            response_time = end_time - start_time
            self.response_times.append(response_time)
            self.total_detection_time += response_time
            self.detection_count += 1
            
            return high_confidence_events
    
    # ADD THIS METHOD
    def get_response_time_stats(self) -> Dict[str, float]:
        """Get evolution detector response time statistics"""
        if not self.response_times:
            return {
                'mean_response_time': 0.0,
                'total_time': 0.0,
                'detection_count': 0,
                'min_response_time': 0.0,
                'max_response_time': 0.0
            }
        
        return {
            'mean_response_time': np.mean(self.response_times),
            'total_time': self.total_detection_time,
            'detection_count': self.detection_count,
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times)
        }
    
    def _initialize_tracking(self, segments: List[SegmentData]):
        """Initialize cluster tracking"""
        for segment in segments:
            self.cluster_tracking[segment.segment_id] = {
                'birth_time': datetime.now(),
                'last_seen': datetime.now(),
                'size_history': [segment.size],
                'stability_history': [segment.stability_score]
            }
    
    def _update_cluster_tracking(self, current_segments: List[SegmentData], 
                               previous_segments: List[SegmentData]):
        """Update cluster tracking information"""
        current_ids = {seg.segment_id for seg in current_segments}
        
        for segment in current_segments:
            if segment.segment_id in self.cluster_tracking:
                tracking = self.cluster_tracking[segment.segment_id]
                tracking['last_seen'] = datetime.now()
                tracking['size_history'].append(segment.size)
                tracking['stability_history'].append(segment.stability_score)
                
                if len(tracking['size_history']) > 20:
                    tracking['size_history'] = tracking['size_history'][-10:]
                if len(tracking['stability_history']) > 20:
                    tracking['stability_history'] = tracking['stability_history'][-10:]
            else:
                self.cluster_tracking[segment.segment_id] = {
                    'birth_time': datetime.now(),
                    'last_seen': datetime.now(),
                    'size_history': [segment.size],
                    'stability_history': [segment.stability_score]
                }
     # 2. Update the evolution detector to calculate revenue impact
    def _detect_segment_births(self, current_segments: List[SegmentData], 
                              previous_segments: List[SegmentData]) -> List[EvolutionEvent]:
        """Detect new segments with revenue impact calculation"""
        events = []
        current_ids = {seg.segment_id for seg in current_segments}
        previous_ids = {seg.segment_id for seg in previous_segments}
        
        new_ids = current_ids - previous_ids
        
        # Calculate total previous revenue for context
        total_previous_revenue = sum(seg.behavioral_metrics.get('total_revenue', 0) 
                                   for seg in previous_segments)
        
        for segment_id in new_ids:
            segment = next(seg for seg in current_segments if seg.segment_id == segment_id)
            
            # Calculate revenue impact
            new_revenue = segment.behavioral_metrics.get('total_revenue', 0)
            revenue_per_user = segment.behavioral_metrics.get('avg_revenue_per_user', 0)
            
            event = EvolutionEvent(
                event_id=f"birth_{segment_id}_{int(datetime.now().timestamp())}",
                event_type=EvolutionEventType.SEGMENT_BIRTH,
                segment_id=segment_id,
                timestamp=datetime.now(),
                confidence=min(segment.size / 50.0, 1.0),
                severity=AlertSeverity.MEDIUM,
                details={
                    'new_size': segment.size,
                    'segment_name': segment.segment_name,
                    'primary_device': segment.behavioral_metrics.get('primary_device', 'unknown'),
                    'conversion_rate': segment.behavioral_metrics.get('conversion_rate', 0)
                },
                predicted_impact={
                    'opportunity_value': segment.size * revenue_per_user,
                    'market_share_gain': new_revenue / max(total_previous_revenue, 1) * 100
                },
                affected_users=list(segment.user_ids),
                # Revenue tracking
                revenue_impact=new_revenue,  # Positive gain
                revenue_before=0.0,
                revenue_after=new_revenue,
                revenue_per_user_before=0.0,
                revenue_per_user_after=revenue_per_user
            )
            events.append(event)
        
        return events
    
    def _detect_segment_deaths(self, current_segments: List[SegmentData], 
                              previous_segments: List[SegmentData]) -> List[EvolutionEvent]:
        """Detect segment deaths with revenue loss calculation"""
        events = []
        current_ids = {seg.segment_id for seg in current_segments}
        previous_ids = {seg.segment_id for seg in previous_segments}
        
        disappeared_ids = previous_ids - current_ids
        
        # Calculate total current revenue for context
        total_current_revenue = sum(seg.behavioral_metrics.get('total_revenue', 0) 
                                  for seg in current_segments)
        
        for segment_id in disappeared_ids:
            prev_segment = next((seg for seg in previous_segments 
                               if seg.segment_id == segment_id), None)
            
            if prev_segment:
                # Calculate revenue loss
                lost_revenue = prev_segment.behavioral_metrics.get('total_revenue', 0)
                revenue_per_user = prev_segment.behavioral_metrics.get('avg_revenue_per_user', 0)
                
                event = EvolutionEvent(
                    event_id=f"death_{segment_id}_{int(datetime.now().timestamp())}",
                    event_type=EvolutionEventType.SEGMENT_DEATH,
                    segment_id=segment_id,
                    timestamp=datetime.now(),
                    confidence=min(prev_segment.size / 30.0, 1.0),
                    severity=AlertSeverity.HIGH,
                    details={
                        'previous_size': prev_segment.size,
                        'segment_name': prev_segment.segment_name,
                        'lost_conversion_rate': prev_segment.behavioral_metrics.get('conversion_rate', 0)
                    },
                    predicted_impact={
                        'revenue_lost': lost_revenue,
                        'customers_lost': prev_segment.size,
                        'market_share_loss': lost_revenue / max(total_current_revenue + lost_revenue, 1) * 100
                    },
                    affected_users=list(prev_segment.user_ids),
                    # Revenue tracking
                    revenue_impact=-lost_revenue,  # Negative loss
                    revenue_before=lost_revenue,
                    revenue_after=0.0,
                    revenue_per_user_before=revenue_per_user,
                    revenue_per_user_after=0.0
                )
                events.append(event)
        
        return events
    def _detect_size_changes(self, current_segments: List[SegmentData], 
                               previous_segments: List[SegmentData]) -> List[EvolutionEvent]:
        """Detect segment changes with user movement tracking"""
        events = []
        
        current_dict = {seg.segment_id: seg for seg in current_segments}
        previous_dict = {seg.segment_id: seg for seg in previous_segments}
        
        common_ids = set(current_dict.keys()) & set(previous_dict.keys())
        
        for segment_id in common_ids:
            current_seg = current_dict[segment_id]
            previous_seg = previous_dict[segment_id]
            
            # Calculate user count changes
            current_users = current_seg.size
            previous_users = previous_seg.size
            user_change = current_users - previous_users
            print(f"DEBUG SIZE: {segment_id} | {previous_users} -> {current_users} | change: {user_change} | %: {abs(user_change)/max(previous_users,1)*100:.1f}%")

            # Calculate revenue changes
            current_revenue = current_seg.behavioral_metrics.get('total_revenue', 0)
            previous_revenue = previous_seg.behavioral_metrics.get('total_revenue', 0)
            
            if current_revenue == 0:
                current_rev_per_user = current_seg.behavioral_metrics.get('avg_revenue_per_user', 0)
                current_revenue = current_rev_per_user * current_users
            
            if previous_revenue == 0:
                previous_rev_per_user = previous_seg.behavioral_metrics.get('avg_revenue_per_user', 0)
                previous_revenue = previous_rev_per_user * previous_users
            
            revenue_change = current_revenue - previous_revenue
            
            current_rev_per_user = current_seg.behavioral_metrics.get('avg_revenue_per_user', 0)
            previous_rev_per_user = previous_seg.behavioral_metrics.get('avg_revenue_per_user', 0)
            
            # Check for significant changes
            size_change = abs(user_change)
            relative_size_change = size_change / max(previous_users, 1)
            relative_revenue_change = abs(revenue_change) / max(previous_revenue, 1) if previous_revenue > 0 else 0
            
            # Trigger event if significant change
            vara=0
            if relative_size_change > 0.1:
                print(f"DEBUG: CREATING EVENT for {segment_id}: size_change={size_change}, rel_change={relative_size_change:.3f}")

                vara=1
                
                if relative_revenue_change > 0.4:
                    severity = AlertSeverity.CRITICAL

                elif relative_revenue_change > 0.3:
                    severity = AlertSeverity.HIGH
                elif relative_revenue_change > 0.2:
                    severity=AlertSeverity.MEDIUM
                else:
                    severity=AlertSeverity.LOW

                
                event = EvolutionEvent(
                    event_id=f"change_{segment_id}_{int(datetime.now().timestamp())}",
                    event_type=EvolutionEventType.SIZE_CHANGE,
                    segment_id=segment_id,
                    timestamp=datetime.now(),
                    confidence=min(relative_size_change + relative_revenue_change, 1.0),
                    severity=severity,
                    details={
                        'previous_size': previous_users,
                        'current_size': current_users,
                        'size_change': user_change,
                        'user_movement': 'gained' if user_change > 0 else 'lost',
                        'segment_name': current_seg.segment_name,
                        'size_change_percent': relative_size_change * 100,
                        'revenue_change_percent': relative_revenue_change * 100
                    },
                    predicted_impact={
                        'revenue_change': revenue_change,
                        'market_share_change': relative_size_change,
                        'per_user_value_change': current_rev_per_user - previous_rev_per_user
                    },
                    affected_users=list(current_seg.user_ids),
                    revenue_impact=revenue_change,
                    revenue_before=previous_revenue,
                    revenue_after=current_revenue,
                    revenue_per_user_before=previous_rev_per_user,
                    revenue_per_user_after=current_rev_per_user
                )
                events.append(event)
            if  size_change>50 and vara==0:
            
                if size_change> 200:
                    severity = AlertSeverity.CRITICAL
            
                elif size_change>100:
                    severity = AlertSeverity.HIGH
                else:
                    severity=AlertSeverity.MEDIUM
            
            
                event = EvolutionEvent(
                    event_id=f"change_{segment_id}_{int(datetime.now().timestamp())}",
                    event_type=EvolutionEventType.SIZE_CHANGE,
                    segment_id=segment_id,
                    timestamp=datetime.now(),
                    confidence=min(relative_size_change + relative_revenue_change, 1.0),
                    severity=severity,
                    details={
                        'previous_size': previous_users,
                        'current_size': current_users,
                        'size_change': user_change,
                        'user_movement': 'gained' if user_change > 0 else 'lost',
                        'segment_name': current_seg.segment_name,
                        'size_change_percent': relative_size_change * 100,
                        'revenue_change_percent': relative_revenue_change * 100
                    },
                    predicted_impact={
                        'revenue_change': revenue_change,
                        'market_share_change': relative_size_change,
                        'per_user_value_change': current_rev_per_user - previous_rev_per_user
                    },
                    affected_users=list(current_seg.user_ids),
                    revenue_impact=revenue_change,
                    revenue_before=previous_revenue,
                    revenue_after=current_revenue,
                    revenue_per_user_before=previous_rev_per_user,
                    revenue_per_user_after=current_rev_per_user
                )
                events.append(event)
            
        for event in events:
           #einstein
            #if event.event_type == EvolutionEventType.SIZE_CHANGE and event.confidence >= self.min_confidence:
            if event.event_type == EvolutionEventType.SIZE_CHANGE :

                segment = next((seg for seg in current_segments if seg.segment_id == event.segment_id), None)
                if segment:
                    self._trigger_immediate_size_change_notification(event, segment)
        return events
        
class FixedOnlineMonitoringSystem:
    """Fixed monitoring system with immediate notifications"""
    
    def __init__(self, user_generation_rate=20.0, monitoring_interval=2):
        self.user_generation_rate = user_generation_rate
        self.monitoring_interval = monitoring_interval
        
        # Core components
        self.user_generator = RealTimeUserGenerator()
        self.profile_manager = UserProfileManager()
        self.clustering_engine = FixedOnlineClusteringEngine()
        self.evolution_detector = OnlineSegmentEvolutionDetector()
        
        # State management
        self.current_segments = []
        self.previous_segments = []
        self.is_running = False
        self.activity_queue = queue.Queue()
        self._lock = threading.RLock()
        
        # Threading
        self.generation_thread = None
        self.processing_thread = None
        self.monitoring_thread = None
        
        # Statistics tracking
        self.stats = {
            'total_activities_generated': 0,
            'total_clustering_operations': 0,
            'new_user_assignments': 0,
            'existing_user_updates': 0,
            'total_evolution_events': 0,
            'immediate_notifications_sent': 0,  # births/deaths only
            'size_change_notifications_sent': 0,  # ADD THIS LINE
            'last_monitoring_time': None
        }
    def add_notification_callback(self, callback: ImmediateNotificationCallback):
        """Add notification callback for immediate alerts"""
        self.evolution_detector.add_notification_callback(callback)
        logger.info(f"Added notification callback to monitoring system: {type(callback).__name__}")
    
    def remove_notification_callback(self, callback: ImmediateNotificationCallback):
        """Remove notification callback"""
        self.evolution_detector.remove_notification_callback(callback)
        logger.info(f"Removed notification callback from monitoring system: {type(callback).__name__}")
    
    def start_monitoring(self):
        """Start the online monitoring system"""
        with self._lock:
            if self.is_running:
                logger.warning("System is already running")
                return
            
            self.is_running = True
            
            logger.info("Starting Online Real-Time Clustering System with IMMEDIATE NOTIFICATIONS")
            logger.info(f"User generation rate: {self.user_generation_rate} users/second")
            logger.info(f"Monitoring interval: {self.monitoring_interval} seconds")
            logger.info(f"Notification callbacks registered: {len(self.evolution_detector.notification_callbacks)}")
            
            # Start threads
            self.generation_thread = threading.Thread(target=self._user_generation_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._online_processing_loop, daemon=True)
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            
            self.generation_thread.start()
            self.processing_thread.start()
            self.monitoring_thread.start()
            
            logger.info("All threads started successfully")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("Stopping online monitoring system...")
        with self._lock:
            self.is_running = False
        
        # Wait for threads to finish
        for thread in [self.generation_thread, self.processing_thread, self.monitoring_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Online monitoring system stopped")
    
    def _user_generation_loop(self):
        """Generate user activities continuously"""
        logger.info("Starting user generation loop")
        acti=0
        while self.is_running:
            try:
                activity = self.user_generator.generate_new_user_activity()
                self.activity_queue.put(activity)
                acti+=1
                if acti%1000==0:
                    
                    print("acti "+str(acti))
                    
                with self._lock:
                    self.stats['total_activities_generated'] += 1
                
                #time.sleep(1.0 / self.user_generation_rate)
               
                if acti%10==0:
                    time.sleep(1.0/50)

            except Exception as e:
                logger.error(f"Error in user generation: {e}")
                time.sleep(1)
    
    def _online_processing_loop(self):
        """Process activities with online clustering and immediate notifications"""
        logger.info("Starting online processing loop with immediate evolution detection")
        
        while self.is_running:
            try:
                processed_count = 0
                while not self.activity_queue.empty() and processed_count < 130:
                    activity = self.activity_queue.get()
                    
                    profile = self.profile_manager.process_user_activity(activity)
                    
                    was_new_user = profile.user_id not in self.clustering_engine.online_kmeans._user_to_cluster
                    
                    cluster_id, current_segments = self.clustering_engine.process_user_profile(profile)
                    
                    # Check for immediate changes and trigger notifications
                    segments_changed = False
                    previous_segments_for_detection = None
                    
                    with self._lock:
                        self.stats['total_clustering_operations'] += 1
                        if was_new_user:
                            self.stats['new_user_assignments'] += 1
                        else:
                            self.stats['existing_user_updates'] += 1
                        
                        # CRITICAL: Capture previous state BEFORE updating current_segments
                        previous_segments_for_detection = self._safe_copy_segments(self.current_segments)
                        
                        # Detect cluster count changes immediately
                        if len(current_segments) != len(self.current_segments):
                            logger.info(f"IMMEDIATE: Cluster count changed: {len(self.current_segments)} -> {len(current_segments)}")
                            segments_changed = True
                        
                        # Now update current segments
                        self.current_segments = current_segments
                    
                    # Trigger immediate evolution detection if segments changed
                    if segments_changed and current_segments and previous_segments_for_detection:
                        try:
                            current_segments_copy = self._safe_copy_segments(current_segments)
                            
                            print(f"DEBUG: Triggering immediate detection")
                            print(f"DEBUG: Previous count: {len(previous_segments_for_detection)}")
                            print(f"DEBUG: Current count: {len(current_segments_copy)}")
                            
                            immediate_events = self.evolution_detector.detect_evolution_events(
                                current_segments_copy, previous_segments_for_detection
                            )
                            
                            if immediate_events:
                                with self._lock:
                                    # Count births and deaths
                                    immediate_notifications = [e for e in immediate_events 
                                                             if e.event_type in [EvolutionEventType.SEGMENT_BIRTH, EvolutionEventType.SEGMENT_DEATH]]
                                    self.stats['immediate_notifications_sent'] += len(immediate_notifications)
                                    
                                    # ADD THIS: Count size changes
                                    size_change_notifications = [e for e in immediate_events 
                                                               if e.event_type == EvolutionEventType.SIZE_CHANGE]
                                    self.stats['size_change_notifications_sent'] += len(size_change_notifications)
                                
                                logger.info(f"IMMEDIATE: Triggered {len(immediate_events)} evolution events during processing!")

                                
                        except Exception as e:
                            logger.error(f"Error in immediate evolution detection: {e}")
                    
                    processed_count += 1
                
                if processed_count == 0:
                    time.sleep(0.05)
                    
            except Exception as e:
                logger.error(f"Error in online processing: {e}")
                time.sleep(1)
    
    def _monitoring_loop(self):
        """Monitor for evolution events with immediate notifications"""
        logger.info(f"Starting monitoring loop with immediate notifications (every {self.monitoring_interval}s)")
        
        while self.is_running:
            try:
                print("MI")
                time.sleep(self.monitoring_interval)
                
                with self._lock:
                    current_segments_copy = self._safe_copy_segments(self.current_segments)
                    previous_segments_copy = self._safe_copy_segments(self.previous_segments)
                
                if current_segments_copy and len(current_segments_copy) > 0:
                    events = self.evolution_detector.detect_evolution_events(
                        current_segments_copy, previous_segments_copy
                    )
                    
                    with self._lock:
                        self.stats['total_evolution_events'] += len(events)
                        # Count immediate notifications (births and deaths)
                        immediate_events = [e for e in events if e.event_type in [EvolutionEventType.SEGMENT_BIRTH, EvolutionEventType.SEGMENT_DEATH]]
                        self.stats['immediate_notifications_sent'] += len(immediate_events)
                        
                        # ADD THIS: Count size changes
                        size_change_events = [e for e in events if e.event_type == EvolutionEventType.SIZE_CHANGE]
                        self.stats['size_change_notifications_sent'] += len(size_change_events)
                        
                        self.stats['last_monitoring_time'] = datetime.now()
                        self.previous_segments = current_segments_copy
                    
                    if events:
                        logger.info(f"Detected {len(events)} evolution events!")
                        for event in events:
                            self._log_evolution_event(event)
                    else:
                        logger.debug("No significant evolution events detected")
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _safe_copy_segments(self, segments: List[SegmentData]) -> List[SegmentData]:
        """Create safe copies of segments without threading objects"""
        segment_copies = []
        for segment in segments:
            segment_copy = SegmentData(
                segment_id=segment.segment_id,
                segment_name=segment.segment_name,
                user_ids=segment.user_ids.copy(),
                centroid=segment.centroid.copy(),
                timestamp=segment.timestamp,
                size=segment.size,
                behavioral_metrics=segment.behavioral_metrics.copy(),
                stability_score=segment.stability_score,
                growth_rate=segment.growth_rate,
                cohesion_score=segment.cohesion_score
            )
            segment_copies.append(segment_copy)
        return segment_copies
    
    # 4. Add revenue tracking to the monitoring loop
    def _log_evolution_event(self, event: EvolutionEvent):
        """Log evolution events with user movement details"""
        severity_colors = {
            AlertSeverity.CRITICAL: "RED",
            AlertSeverity.HIGH: "ORANGE", 
            AlertSeverity.MEDIUM: "YELLOW",
            AlertSeverity.LOW: "GREEN"
        }
        
        color = severity_colors.get(event.severity, "WHITE")
        
        if event.event_type == EvolutionEventType.SEGMENT_BIRTH:
            logger.info(f"{color} BIRTH - {event.segment_id} | +{event.details['new_size']} users | +${event.revenue_impact:,.2f} revenue")
        elif event.event_type == EvolutionEventType.SEGMENT_DEATH:
            logger.info(f"{color} DEATH - {event.segment_id} | -{event.details['previous_size']} users | -${abs(event.revenue_impact):,.2f} revenue")
        elif event.event_type == EvolutionEventType.SIZE_CHANGE:
            user_change = event.details['size_change']
            change_symbol = "+" if user_change >= 0 else ""
            revenue_symbol = "+" if event.revenue_impact >= 0 else ""
            logger.info(f"{color} CHANGE - {event.segment_id} | {change_symbol}{user_change} users | {revenue_symbol}${event.revenue_impact:,.2f} revenue")
            logger.info(f"   Users: {event.details['previous_size']} → {event.details['current_size']} ({event.details['user_movement']} {abs(user_change)})")
        
        logger.info(f"   Revenue: ${event.revenue_before:,.2f} → ${event.revenue_after:,.2f}")
        logger.info(f"   Per User: ${event.revenue_per_user_before:.2f} → ${event.revenue_per_user_after:.2f}")

    def get_system_status(self) -> Dict:
        """Get system status with drift analytics"""
        total_revenue = 0.0
        total_users = 0
        total_silhouette = 0.0
        valid_segments = 0
        
        for segment in self.current_segments:
            segment_revenue = segment.behavioral_metrics.get('total_revenue', 0)
            total_revenue += segment_revenue
            total_users += segment.size
            if hasattr(segment, 'silhouette_score'):
                total_silhouette += segment.silhouette_score
                valid_segments += 1
        
        avg_revenue_per_user = total_revenue / max(total_users, 1)
        avg_silhouette_score = total_silhouette / max(valid_segments, 1)
        
        with self._lock:
            unique_users = self.profile_manager.get_user_count()
            clustering_stats = self.clustering_engine.get_clustering_stats()
            
            # Get evolution detector response time stats
            response_stats = self.evolution_detector.get_response_time_stats()
            
            # ADD DRIFT STATISTICS
            drift_stats = self.clustering_engine.online_kmeans.get_drift_stats()
            
            return {
                'system_status': 'running' if self.is_running else 'stopped',
                'clustering_mode': 'FIXED_ONE_PASS_ONLINE',
                'immediate_notifications': {
                    'enabled': len(self.evolution_detector.notification_callbacks) > 0,
                    'callback_count': len(self.evolution_detector.notification_callbacks),
                    'notifications_sent': self.stats['immediate_notifications_sent']
                },
                'current_stats': self.stats.copy(),
                'clustering_moves': {
                    'total_moves': self.clustering_engine.online_kmeans.total_clustering_moves,
                    'moves_per_user': self.clustering_engine.online_kmeans.total_clustering_moves / max(unique_users, 1)
                },
                # ADD COMPREHENSIVE DRIFT SECTION
                'user_drift_analytics': drift_stats,
                'evolution_detector_performance': response_stats,
                'user_accounting': {
                    'unique_users_total': unique_users,
                    'assigned_users': clustering_stats['assigned_users'],
                    'unassigned_users': clustering_stats['unassigned_users'],
                    'total_tracked': clustering_stats['total_tracked_users'],
                    'accounting_complete': clustering_stats['accounting_complete']
                },
                'cluster_accounting': {
                    'total_clusters': clustering_stats['total_clusters'],
                    'valid_clusters': clustering_stats['valid_clusters'],
                    'small_clusters': clustering_stats['small_clusters'],
                    'cluster_sizes': clustering_stats['cluster_sizes'],
                    'total_cluster_users': clustering_stats['total_cluster_users'],
                    'min_size_threshold': clustering_stats['min_cluster_size_threshold']
                },
                'current_segments': len(self.current_segments),
                'queue_size': self.activity_queue.qsize(),
                'total_revenue': total_revenue,
                'avg_revenue_per_user': avg_revenue_per_user,
                'avg_silhouette_score': avg_silhouette_score,
                'revenue_summary': {
                    'total_revenue': total_revenue,
                    'total_users': total_users,
                    'avg_revenue_per_user': avg_revenue_per_user
                },
                         # ... existing fields ...
            'immediate_notifications': {
                'enabled': len(self.evolution_detector.notification_callbacks) > 0,
                'callback_count': len(self.evolution_detector.notification_callbacks),
                'notifications_sent': self.stats['immediate_notifications_sent'],
                'size_change_notifications_sent': self.stats['size_change_notifications_sent']  # ADD THIS LINE
            },
                'segment_details': [
                    {
                        'segment_id': seg.segment_id,
                        'name': seg.segment_name,
                        'size': seg.size,
                        'stability_score': seg.stability_score,
                        'silhouette_score': getattr(seg, 'silhouette_score', 0.0),
                        'key_metrics': seg.behavioral_metrics
                    }
                    for seg in self.current_segments
                ]
            }

# ENHANCED: Demo with Immediate Notifications
class FixedDemo:
    """Demo for the fixed online clustering system with immediate notifications"""
    
    def __init__(self):
        self.system = FixedOnlineMonitoringSystem(
            user_generation_rate=40.0,
            monitoring_interval=2
        )
        self._setup_notification_callbacks()
    
    def _setup_notification_callbacks(self):
        """Setup various notification callbacks"""
        # Console notifications (always enabled)
        console_callback = ConsoleNotificationCallback()
        self.system.add_notification_callback(console_callback)
        
        # Log file notifications
        log_callback = LogFileNotificationCallback("segment_events.log")
        self.system.add_notification_callback(log_callback)
        
        # Example: Email notifications (uncomment and configure as needed)
        # email_callback = EmailNotificationCallback(
        #     smtp_server="smtp.gmail.com",
        #     smtp_port=587,
        #     username="your_email@gmail.com",
        #     password="your_password",
        #     to_emails=["admin@company.com"],
        #     from_email="alerts@company.com"
        # )
        # self.system.add_notification_callback(email_callback)
        
        # Example: Webhook notifications (uncomment and configure as needed)
        # webhook_callback = WebhookNotificationCallback(
        #     webhook_url="https://your-webhook-url.com/segment-alerts"
        # )
        # self.system.add_notification_callback(webhook_callback)
    
    def run_demo(self, duration_minutes=2):
        """Run demo with immediate notifications"""
        print("Online Clustering Demo with IMMEDIATE NOTIFICATIONS")
        print("=" * 60)
        print("Segment births and deaths will trigger instant alerts!")
        print(f"Generation rate: {self.system.user_generation_rate} users/second")
        print(f"Monitoring every {self.system.monitoring_interval} seconds")
        print(f"Notification callbacks: {len(self.system.evolution_detector.notification_callbacks)}")
        print()
        
        self.system.start_monitoring()
        
        try:
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            last_status_time = start_time
            
            while time.time() < end_time:
                current_time = time.time()
                
                if current_time - last_status_time >= 20:
                    self._print_status_with_notifications()
                    last_status_time = current_time
                
                time.sleep(5)
            
            print("\nDemo completed!")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted")
        finally:
            self.system.stop_monitoring()
            self._print_final_notification_stats()
    def analyze_top_drifting_users(self, top_n=10):
        """Analyze and display top drifting users"""
        system = self.system
        drift_stats = system.clustering_engine.online_kmeans.get_drift_stats()
        
        if drift_stats['unique_drifting_users'] == 0:
            print("No drifting users found.")
            return
        
        # Get top drifting users
        user_drift_counts = system.clustering_engine.online_kmeans.user_drift_count
        top_drifters = sorted(user_drift_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nTOP {min(top_n, len(top_drifters))} DRIFTING USERS:")
        print("-" * 50)
        
        for user_id, drift_count in top_drifters:
            pattern = system.clustering_engine.online_kmeans.get_user_drift_pattern(user_id)
            cluster_path = " -> ".join(map(str, pattern['cluster_history'][-5:]))  # Last 5 clusters
            print(f"User {user_id[:12]}:")
            print(f"  Drift Count: {drift_count}")
            print(f"  Drift Level: {pattern['drift_level']}")
            print(f"  Cluster Path: ...{cluster_path}")
            print(f"  Unique Clusters: {pattern['unique_clusters_visited']}")
            print()
    def _print_status_with_notifications(self):
        """Print system status including drift analytics"""
        status = self.system.get_system_status()
        
        print("\nReal-Time Clustering Status with DRIFT ANALYTICS")
        print("-" * 60)
        stats = status["current_stats"]

        print(f"Activities generated: {stats['total_activities_generated']}")
        print(f"Clustering operations: {stats['total_clustering_operations']}")
        print(f"Unique users: {stats['new_user_assignments']}")
        print(f"returning users: {stats['existing_user_updates']}")
        print(f"Queue size: {status['queue_size']}")
        
        clustering_moves = status['clustering_moves']
        print(f"Total drifting moves: {clustering_moves['total_moves']}")
        print(f"Moves per user: {clustering_moves['moves_per_user']:.2f}")
        
        # ADD DRIFT ANALYTICS DISPLAY
        drift_stats = status['user_drift_analytics']
        print(f"\nUSER DRIFT ANALYTICS:")
        print(f"  Total drift events: {drift_stats['total_drift_events']}")
        print(f"  Drifting users: {drift_stats['unique_drifting_users']}")
        print(f"  Stable users: {drift_stats['stable_users']}")
        print(f"  Avg drift per user: {drift_stats['avg_drift_per_user']:.2f}")
        print(f"  Max drift per user: {drift_stats['max_drift_per_user']}")
        print(f"  High drift users (3+ moves): {drift_stats['high_drift_users']}")
        print(f"  Avg drift distance: {drift_stats['avg_drift_distance']:.3f}")
        print(f"  Drift rate: {drift_stats['drift_rate']:.3f}")
        
        print(f"  Drift Distribution:")
        for pattern, count in drift_stats['drift_distribution'].items():
            print(f"    {pattern}: {count} users")
        
        perf_stats = status['evolution_detector_performance']
        print(f"\nPerformance:")
        print(f"  Evolution detector mean time: {perf_stats['mean_response_time']:.4f}s")
        print(f"  Evolution detector total time: {perf_stats['total_time']:.2f}s")
        print(f"  Average silhouette score: {status['avg_silhouette_score']:.3f}")
        
        user_acc = status['user_accounting']
        cluster_acc = status['cluster_accounting']
        notifications = status['immediate_notifications']
        
        print(f"\nUSER ACCOUNTING:")
        print(f"  Total Users: {user_acc['unique_users_total']}")
        print(f"  In Valid Clusters: {user_acc['assigned_users']}")
        print(f"  In Small Clusters: {user_acc['unassigned_users']}")
        
        print(f"\nCLUSTER STATUS:")
        print(f"  Valid Clusters: {cluster_acc['valid_clusters']}")
        print(f"  Small Clusters: {cluster_acc['small_clusters']}")
        print(f"  Cluster Sizes: {cluster_acc['cluster_sizes']}")
        
        print(f"\nIMMEDIATE NOTIFICATIONS:")
        print(f"Birth/Death Notifications: {notifications['notifications_sent']}")
        print(f"Size Change Notifications: {notifications['size_change_notifications_sent']}")  # ADD THIS LINE
        print(f"Total Notifications: {notifications['notifications_sent'] + notifications['size_change_notifications_sent']}")  # ADD THIS LINE
        print(f"Notification Callbacks Used: {notifications['callback_count']}")
        if status['segment_details']:
            print(f"\nACTIVE SEGMENTS:")
            for segment in status['segment_details'][:]:
                sil_score = segment['silhouette_score']
                print(f"  • {segment['name']}: {segment['size']} users (sil: {sil_score:.3f})")
        print()
    def _print_final_notification_stats(self):
        """Print final notification statistics"""
        status = self.system.get_system_status()
        notifications = status['immediate_notifications']
        
        print("\nFINAL NOTIFICATION RESULTS")
        print("=" * 40)
        print(f"Birth/Death Notifications: {notifications['notifications_sent']}")
        print(f"Size Change Notifications: {notifications['size_change_notifications_sent']}")  # ADD THIS LINE
        print(f"Total Notifications: {notifications['notifications_sent'] + notifications['size_change_notifications_sent']}")  # ADD THIS LINE
        print(f"Notification Callbacks Used: {notifications['callback_count']}")
        #print(f"Total Evolution Events: {status['current_stats']['total_evolution_events']}")
       # print(f"Immediate Notifications Sent: {notifications['notifications_sent']}")
        #print(f"Notification Callbacks Used: {notifications['callback_count']}")
       # print(f"Total Evolution Events: {status['current_stats']['total_evolution_events']}")
        
        print(f"\nFINAL SEGMENTS:")
        for segment in status['segment_details']:
            print(f"  • {segment['name']}: {segment['size']} users")
        
        print("\nNOTIFICATION FEATURES:")
        print("  ✓ Console alerts for births/deaths")
        print("  ✓ Log file tracking")
        print("  ✓ Email notifications (configurable)")
        print("  ✓ Webhook integration (configurable)")
        print("  ✓ Thread-safe immediate triggering")
        
        if notifications['notifications_sent'] > 0:
            print(f"\n✓ SUCCESS: {notifications['notifications_sent']} immediate notifications delivered!")
        else:
            print(f"\n• No segment births/deaths detected during demo period")
        
        print("\nCheck 'segment_events.log' for detailed event history!")

# Usage Examples and Configuration
class NotificationExamples:
    """Examples of how to configure different notification types"""
    
    @staticmethod
    def setup_console_notifications(system: FixedOnlineMonitoringSystem):
        """Setup basic console notifications"""
        console_callback = ConsoleNotificationCallback()
        system.add_notification_callback(console_callback)
        return console_callback
    
    @staticmethod
    def setup_email_notifications(system: FixedOnlineMonitoringSystem, 
                                 email_config: dict):
        """Setup email notifications"""
        email_callback = EmailNotificationCallback(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            username=email_config['username'],
            password=email_config['password'],
            to_emails=email_config['to_emails'],
            from_email=email_config.get('from_email')
        )
        system.add_notification_callback(email_callback)
        return email_callback
    
    @staticmethod
    def setup_webhook_notifications(system: FixedOnlineMonitoringSystem,
                                   webhook_url: str, headers: dict = None):
        """Setup webhook notifications"""
        webhook_callback = WebhookNotificationCallback(
            webhook_url=webhook_url,
            headers=headers
        )
        system.add_notification_callback(webhook_callback)
        return webhook_callback
    
    @staticmethod
    def setup_log_file_notifications(system: FixedOnlineMonitoringSystem,
                                    log_file_path: str = "segment_events.log"):
        """Setup log file notifications"""
        log_callback = LogFileNotificationCallback(log_file_path)
        system.add_notification_callback(log_callback)
        return log_callback

# Custom notification callback example
class SlackNotificationCallback(ImmediateNotificationCallback):
    """Custom Slack notification callback (example)"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def _send_slack_message(self, message: str):
        """Send message to Slack"""
        try:
            import requests
            payload = {
                "channel": self.channel,
                "text": message,
                "username": "Segment Monitor"
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    
    def on_segment_birth(self, event: EvolutionEvent, segment: SegmentData):
        print(f"\n*** SEGMENT BIRTH ALERT! ***")
        print(f"   Segment: {segment.segment_name}")
        print(f"   Size: {segment.size} users")
        print(f"   💰 REVENUE GAIN: ${event.revenue_impact:,.2f}")
        print(f"   💰 Revenue per User: ${event.revenue_per_user_after:.2f}")
        print(f"   📈 Market Opportunity: ${event.predicted_impact.get('opportunity_value', 0):,.2f}")
        print(f"   📊 Market Share Gain: {event.predicted_impact.get('market_share_gain', 0):.1f}%")
        print(f"   Timestamp: {event.timestamp}")
        print("-" * 60)
    
    def on_segment_death(self, event: EvolutionEvent, last_known_segment: SegmentData):
        print(f"\n*** SEGMENT DEATH ALERT! ***")
        print(f"   Segment: {last_known_segment.segment_name}")
        print(f"   Lost Size: {last_known_segment.size} users")
        print(f"   💸 REVENUE LOST: ${abs(event.revenue_impact):,.2f}")
        print(f"   💸 Revenue per User Lost: ${event.revenue_per_user_before:.2f}")
        print(f"   📉 Market Share Lost: {event.predicted_impact.get('market_share_loss', 0):.1f}%")
        print(f"   ⚠️  Customers Lost: {event.predicted_impact.get('customers_lost', 0)}")
        print(f"   Timestamp: {event.timestamp}")
        print("-" * 60)
# Run the complete system with immediate notifications
if __name__ == "__main__":
    print("Real-Time Clustering System with IMMEDIATE NOTIFICATIONS")
    print("=" * 65)
    print("Choose option:")
    print("1. Quick Demo with Immediate Notifications (2 minutes)")
    print("2. Extended Demo with Immediate Notifications (5 minutes)")
    print("3. Custom Notification Setup")
    print("4. Test Notification System Only")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("200 events/user activities per second")
        demo = FixedDemo()
        demo.run_demo(duration_minutes=1)
        demo._print_status_with_notifications()
        
    elif choice == "2":
        demo = FixedDemo()
        demo.run_demo(duration_minutes=5)
        
    elif choice == "3":
        print("\nCustom Notification Setup")
        print("-" * 30)
        
        system = FixedOnlineMonitoringSystem(user_generation_rate=25.0)
        
        # Setup notifications based on user input
        use_console = input("Enable console notifications? (y/n): ").lower() == 'y'
        if use_console:
            NotificationExamples.setup_console_notifications(system)
        
        use_log = input("Enable log file notifications? (y/n): ").lower() == 'y'
        if use_log:
            log_file = input("Log file path (default: segment_events.log): ").strip()
            if not log_file:
                log_file = "segment_events.log"
            NotificationExamples.setup_log_file_notifications(system, log_file)
        
        use_webhook = input("Enable webhook notifications? (y/n): ").lower() == 'y'
        if use_webhook:
            webhook_url = input("Webhook URL: ").strip()
            if webhook_url:
                NotificationExamples.setup_webhook_notifications(system, webhook_url)
        
        duration = int(input("Demo duration in minutes (default: 3): ") or "3")
        
        print(f"\nStarting custom demo for {duration} minutes...")
        demo = FixedDemo()
        demo.system = system
        demo.run_demo(duration_minutes=duration)
        
    elif choice == "4":
        print("\nTesting Notification System...")
        
        # Create test system
        system = FixedOnlineMonitoringSystem(user_generation_rate=30.0, monitoring_interval=10)
        
        # Add all notification types
        NotificationExamples.setup_console_notifications(system)
        NotificationExamples.setup_log_file_notifications(system)
        
        print("Running accelerated test to trigger notifications...")
        system.start_monitoring()
        
        try:
            time.sleep(60)  # Run for 1 minute with high generation rate
        finally:
            system.stop_monitoring()
            
        print("Notification test completed! Check console output and segment_events.log")
        
    else:
        print("Invalid choice. Running default demo...")
        demo = FixedDemo()
        demo.run_demo(duration_minutes=2)