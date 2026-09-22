/*
 * faber
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * faber is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * faber is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with faber.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

use chrono::{Days, SecondsFormat, Utc};
use dirs;
use log::{debug, trace, warn};
use reqwest::StatusCode;
use reqwest::blocking::{Client, Response};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue, RETRY_AFTER, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Issue {
    pub url: String,
    pub repository_url: String,
    pub labels_url: String,
    pub comments_url: String,
    pub events_url: String,
    pub html_url: String,
    pub id: u64,
    pub node_id: String,
    pub number: u64,
    pub title: String,
    pub user: User,

    pub labels: Vec<Type>,
    pub state: Option<String>,
    pub locked: bool,
    pub assignee: Option<User>,
    pub assignees: Vec<User>,
    pub milestone: Option<Value>,
    pub comments: Option<u64>,
    pub created_at: String,
    pub updated_at: String,
    pub closed_at: Option<String>,

    pub author_association: String,
    // Renamed because 'type' is a Rust keyword
    #[serde(rename = "type")]
    pub issue_type: Option<Type>,
    pub sub_issues_summary: Option<SubIssuesSummary>,
    pub active_lock_reason: Option<String>,
    pub body: Option<String>,
    pub closed_by: Option<Value>,
    pub reactions: Option<Reactions>,
    pub timeline_url: Option<String>,
    pub performed_via_github_app: Option<String>,
    pub state_reason: Option<String>,

    // Fields specific to Pull Requests (optional in the general issue list)
    #[serde(default)] // Treat missing field as None
    pub draft: Option<bool>,
    #[serde(default)] // Treat missing field as None
    pub pull_request: Option<PullRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Comment {
    pub url: String,
    pub html_url: String,
    pub issue_url: String,
    pub id: u64,
    pub node_id: String,
    pub user: User,
    pub created_at: String,
    pub updated_at: String,
    pub author_association: String,
    pub body: String,
    pub reactions: Option<Reactions>,
    pub performed_via_github_app: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Type {
    pub id: u64,
    pub node_id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub color: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub is_enabled: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct User {
    pub login: String,
    pub id: u64,
    pub node_id: String,
    pub avatar_url: String,
    pub gravatar_id: String,
    pub url: String,
    pub html_url: String,
    pub followers_url: String,
    pub following_url: String,
    pub gists_url: String,
    pub starred_url: String,
    pub subscriptions_url: String,
    pub organizations_url: String,
    pub repos_url: String,
    pub events_url: String,
    pub received_events_url: String,
    // Renamed because 'type' is a Rust keyword
    #[serde(rename = "type")]
    pub user_type: String,
    pub user_view_type: String,
    pub site_admin: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SubIssuesSummary {
    pub total: u64,
    pub completed: u64,
    pub percent_completed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Reactions {
    pub url: String,
    pub total_count: u64,
    #[serde(rename = "+1")]
    pub plus1: u64,
    #[serde(rename = "-1")]
    pub minus1: u64,
    pub laugh: u64,
    pub hooray: u64,
    pub confused: u64,
    pub heart: u64,
    pub rocket: u64,
    pub eyes: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PullRequest {
    pub url: String,
    pub html_url: String,
    pub diff_url: String,
    pub patch_url: String,
    pub merged_at: Option<String>,

    pub id: Option<u64>,
    pub node_id: Option<String>,
    pub issue_url: Option<String>,
    pub number: Option<u32>,
    pub state: Option<String>,
    pub locked: Option<bool>,
    pub title: Option<String>,
    pub user: Option<User>,
    pub body: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub closed_at: Option<String>,
    pub merge_commit_sha: Option<String>,
    pub assignee: Option<User>,
    pub assignees: Option<Vec<User>>,
    pub requested_reviewers: Option<Vec<User>>,
    pub requested_teams: Option<Vec<Team>>,
    pub labels: Option<Vec<Label>>,
    pub milestone: Option<Milestone>,
    pub draft: Option<bool>,
    pub commits_url: Option<String>,
    pub review_comments_url: Option<String>,
    pub review_comment_url: Option<String>,
    pub comments_url: Option<String>,
    pub statuses_url: Option<String>,
    pub head: Option<RepoRef>,
    pub base: Option<RepoRef>,
    #[serde(rename = "_links")]
    pub links: Option<Links>,
    pub author_association: Option<String>,
    pub auto_merge: Option<bool>,
    pub active_lock_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Team {
    pub id: Option<String>,
    pub node_id: Option<String>,
    pub url: Option<String>,
    pub html_url: Option<String>,
    pub name: Option<String>,
    pub slug: Option<String>,
    pub description: Option<String>,
    pub privacy: Option<String>,
    pub notication_setting: Option<String>,
    pub permission: Option<String>,
    pub members_url: Option<String>,
    pub repositories_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Label {
    pub id: u64,
    pub node_id: String,
    pub url: String,
    pub name: String,
    pub color: String,
    pub default: bool,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Milestone {
    pub url: String,
    pub html_url: String,
    pub labels_url: String,
    pub id: u64,
    pub node_id: String,
    pub number: u32,
    pub state: String,
    pub title: String,
    pub description: Option<String>,
    pub creator: User,
    pub open_issues: u32,
    pub closed_issues: u32,
    pub created_at: String,
    pub updated_at: String,
    pub closed_at: Option<String>,
    pub due_on: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct RepoRef {
    pub label: Option<String>,
    pub ref_field: Option<String>,
    #[serde(rename = "ref")]
    pub sha: Option<String>,
    pub user: Option<User>,
    pub repo: Option<Repository>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Repository {
    pub id: u64,
    pub node_id: Option<String>,
    pub name: Option<String>,
    pub full_name: Option<String>,
    pub private: bool,
    pub owner: User,
    pub html_url: Option<String>,
    pub description: Option<String>,
    pub fork: bool,
    pub url: Option<String>,
    pub forks_url: Option<String>,
    pub keys_url: Option<String>,
    pub collaborators_url: Option<String>,
    pub teams_url: Option<String>,
    pub hooks_url: Option<String>,
    pub issue_events_url: Option<String>,
    pub events_url: Option<String>,
    pub assignees_url: Option<String>,
    pub branches_url: Option<String>,
    pub tags_url: Option<String>,
    pub blobs_url: Option<String>,
    pub git_tags_url: Option<String>,
    pub git_refs_url: Option<String>,
    pub trees_url: Option<String>,
    pub statuses_url: Option<String>,
    pub languages_url: Option<String>,
    pub stargazers_url: Option<String>,
    pub contributors_url: Option<String>,
    pub subscribers_url: Option<String>,
    pub subscription_url: Option<String>,
    pub commits_url: Option<String>,
    pub git_commits_url: Option<String>,
    pub comments_url: Option<String>,
    pub issue_comment_url: Option<String>,
    pub contents_url: Option<String>,
    pub compare_url: Option<String>,
    pub merges_url: Option<String>,
    pub archive_url: Option<String>,
    pub downloads_url: Option<String>,
    pub issues_url: Option<String>,
    pub pulls_url: Option<String>,
    pub milestones_url: Option<String>,
    pub notifications_url: Option<String>,
    pub labels_url: Option<String>,
    pub releases_url: Option<String>,
    pub deployments_url: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub pushed_at: Option<String>,
    pub git_url: Option<String>,
    pub ssh_url: Option<String>,
    pub clone_url: Option<String>,
    pub svn_url: Option<String>,
    pub homepage: Option<String>,
    pub size: u64,
    pub stargazers_count: u64,
    pub watchers_count: u64,
    pub language: Option<String>,
    pub has_issues: bool,
    pub has_projects: bool,
    pub has_downloads: bool,
    pub has_wiki: bool,
    pub has_pages: bool,
    pub has_discussions: bool,
    pub forks_count: u64,
    pub mirror_url: Option<String>,
    pub archived: bool,
    pub disabled: bool,
    pub open_issues_count: u64,
    pub license: Option<License>,
    pub allow_forking: bool,
    pub is_template: bool,
    pub web_commit_signoff_required: bool,
    pub topics: Vec<String>,
    pub visibility: Option<String>,
    pub forks: u64,
    pub open_issues: u64,
    pub watchers: u64,
    pub default_branch: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct License {
    pub key: Option<String>,
    pub name: Option<String>,
    pub pdx_id: Option<String>,
    pub url: Option<String>,
    pub node_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Links {
    #[serde(rename = "self")]
    pub self_link: Link,
    pub html: Link,
    pub issue: Link,
    pub comments: Link,
    pub review_comments: Link,
    pub review_comment: Link,
    pub commits: Link,
    pub statuses: Link,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Link {
    pub href: String,
}

// Top-level
pub type Issues = Vec<Issue>;
pub type PullRequests = Vec<PullRequest>;
pub type Comments = Vec<Comment>;

/// Reads the GitHub API key from the file `~/.github/token`.
fn read_github_token() -> Result<String, Box<dyn Error>> {
    let home_dir = dirs::home_dir().ok_or("Could not find home directory")?;
    let token_path = home_dir.join(".github").join("token");

    match std::fs::read_to_string(&token_path) {
        Ok(token) => {
            let token = token.trim().to_string();
            if token.is_empty() {
                warn!("GitHub token file {:?} is empty.", token_path);
                Err("GitHub token file is empty".into())
            } else {
                debug!("Successfully read GitHub token from {:?}", token_path);
                Ok(token)
            }
        }
        Err(e) => {
            debug!(
                "Could not read GitHub token from {:?}: {}. Proceeding without token.",
                token_path, e
            );
            Err(e.into()) // Propagate the error to indicate token wasn't read
        }
    }
}

const MAX_RETRIES: u32 = 5;

const X_RATELIMIT_REMAINING: &str = "x-ratelimit-remaining";
const X_RATELIMIT_RESET: &str = "x-ratelimit-reset";

fn make_request(url: &String) -> Result<Response, Box<dyn Error>> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("faber"));

    if let Ok(token) = read_github_token() {
        debug!("Using GitHub token for API request to {}", url);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token))?,
        );
    } else {
        debug!(
            "Proceeding with API request to {} without GitHub token.",
            url
        );
    }

    let client = Client::new();

    for attempt in 1..=MAX_RETRIES {
        debug!(
            "Sending GET request to {} (Attempt {}/{})",
            url, attempt, MAX_RETRIES
        );

        let response_result = client.get(url).headers(headers.clone()).send();

        match response_result {
            Ok(response) => {
                let status = response.status();
                debug!("Received response with status: {} for url {}", status, url);

                if status.is_success() {
                    trace!("Request to {} completed successfully", url);
                    return Ok(response);
                }

                if (status == StatusCode::FORBIDDEN
                    || status == StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error())
                    && attempt < MAX_RETRIES
                {
                    let response_headers = response.headers().clone();
                    let mut delay_duration = Duration::from_secs(5 * attempt as u64);

                    if status == StatusCode::FORBIDDEN || status == StatusCode::TOO_MANY_REQUESTS {
                        if let Some(retry_after_val) = response_headers.get(RETRY_AFTER) {
                            if let Ok(retry_after_str) = retry_after_val.to_str() {
                                if let Ok(seconds) = retry_after_str.parse::<u64>() {
                                    delay_duration = Duration::from_secs(seconds);
                                    warn!(
                                        "Rate limited (status {}). Retrying after {} seconds (Retry-After header) for URL: {}",
                                        status, seconds, url
                                    );
                                } else {
                                    warn!(
                                        "Rate limited (status {}). Could not parse Retry-After header value as seconds: \'{}\'. Using default backoff for URL: {}",
                                        status, retry_after_str, url
                                    );
                                }
                            }
                        } else if let (Some(remaining_val), Some(reset_val)) = (
                            response_headers.get(X_RATELIMIT_REMAINING),
                            response_headers.get(X_RATELIMIT_RESET),
                        ) {
                            if let (Ok(remaining_str), Ok(reset_str)) =
                                (remaining_val.to_str(), reset_val.to_str())
                            {
                                if let (Ok(remaining), Ok(reset_timestamp_epoch_secs)) =
                                    (remaining_str.parse::<u64>(), reset_str.parse::<u64>())
                                {
                                    if remaining == 0 {
                                        match SystemTime::now().duration_since(UNIX_EPOCH) {
                                            Ok(current_time_since_epoch) => {
                                                let current_timestamp_secs =
                                                    current_time_since_epoch.as_secs();
                                                if reset_timestamp_epoch_secs
                                                    > current_timestamp_secs
                                                {
                                                    let wait_seconds = reset_timestamp_epoch_secs
                                                        - current_timestamp_secs;
                                                    let max_wait_seconds: u64 = 15 * 60;
                                                    let actual_wait_seconds =
                                                        wait_seconds.min(max_wait_seconds);
                                                    delay_duration =
                                                        Duration::from_secs(actual_wait_seconds);
                                                    warn!(
                                                        "Rate limited (status {}, X-RateLimit-Remaining: 0). Retrying after {} seconds (X-RateLimit-Reset: {}) for URL: {}",
                                                        status,
                                                        actual_wait_seconds,
                                                        reset_timestamp_epoch_secs,
                                                        url
                                                    );
                                                    if wait_seconds > max_wait_seconds {
                                                        warn!(
                                                            "Original X-RateLimit-Reset suggested {}s, capped to {}s for URL: {}",
                                                            wait_seconds, max_wait_seconds, url
                                                        );
                                                    }
                                                } else {
                                                    warn!(
                                                        "Rate limited (status {}, X-RateLimit-Remaining: 0), but reset time ({}) is in the past. Using default backoff for URL: {}",
                                                        status, reset_timestamp_epoch_secs, url
                                                    );
                                                }
                                            }
                                            Err(e_time) => {
                                                warn!(
                                                    "Could not get current system time: {}. Using default backoff for URL: {}",
                                                    e_time, url
                                                );
                                            }
                                        }
                                    } else {
                                        warn!(
                                            "Status {} with X-RateLimit-Remaining: {}. This might not be a rate limit. Using default backoff for URL: {}",
                                            status, remaining, url
                                        );
                                    }
                                } else {
                                    warn!(
                                        "Rate limited (status {}). Could not parse X-RateLimit headers as numbers. Using default backoff for URL: {}",
                                        status, url
                                    );
                                }
                            } else {
                                warn!(
                                    "Rate limited (status {}). X-RateLimit headers not valid strings. Using default backoff for URL: {}",
                                    status, url
                                );
                            }
                        } else {
                            warn!(
                                "Rate limited (status {}). No specific guidance headers (Retry-After, X-RateLimit-Reset with Remaining=0). Using default backoff for URL: {}",
                                status, url
                            );
                        }
                    } else if status.is_server_error() {
                        warn!(
                            "Server error (status {}). Retrying with backoff ({}s) for URL: {}",
                            status,
                            delay_duration.as_secs(),
                            url
                        );
                    }

                    debug!(
                        "Sleeping for {:?} before retrying request to {}",
                        delay_duration, url
                    );
                    thread::sleep(delay_duration);
                    continue;
                }

                match response.error_for_status() {
                    Ok(_should_not_happen_if_status_is_error) => Err(format!(
                        "HTTP status {} was not success but error_for_status returned Ok for URL: {}",
                        status, url
                    )),
                    Err(e) => {
                        debug!(
                            "HTTP error status for url {}: {} (after {} attempts)",
                            url, e, attempt
                        );
                        Err(format!("{}", e).into())
                    }
                }?
            }
            Err(e) => {
                if attempt < MAX_RETRIES {
                    let backoff_duration = Duration::from_secs(2 * attempt as u64);
                    warn!(
                        "Request to {} failed: {}. Retrying in {:?} (Attempt {}/{})",
                        url, e, backoff_duration, attempt, MAX_RETRIES
                    );
                    thread::sleep(backoff_duration);
                    continue;
                }
                warn!(
                    "Request to {} failed after {} attempts: {}",
                    url, MAX_RETRIES, e
                );
                return Err(e.into());
            }
        }
    }
    Err(format!("Exhausted {} retries for request to {}", MAX_RETRIES, url).into())
}

/// Makes an HTTP GET request to a standard GitHub URL (not the API).
fn make_github_request(path: &String) -> Result<Response, Box<dyn Error>> {
    let url = format!("https://www.github.com/{}", path);
    debug!("Making GitHub request to: {}", url);

    let response = make_request(&url)?;

    debug!("GitHub request successful: {}", path);
    Ok(response)
}

/// Makes an HTTP GET request to the GitHub API.
fn make_github_api_request(
    path: &String,
    query: Option<&String>,
) -> Result<Response, Box<dyn Error>> {
    let url = match query {
        Some(query) => {
            let full_url = format!("https://api.github.com/{}?{}", path, query);
            debug!("Making GitHub API request with query: {}", full_url);
            full_url
        }
        None => {
            let full_url = format!("https://api.github.com/{}", path);
            debug!("Making GitHub API request: {}", full_url);
            full_url
        }
    };

    let response = make_request(&url)?;

    debug!("GitHub API request successful: {}", path);
    Ok(response)
}

/// Fetches issues from a GitHub repository that have been updated within the last `days`.
pub fn get_github_issues(repo: &String, days: u64) -> Result<Issues, Box<dyn Error>> {
    debug!(
        "Fetching issues from repository {} updated in the last {} days",
        repo, days
    );

    let now = Utc::now();
    let since = now.checked_sub_days(Days::new(days)).unwrap();
    let query = format!("since={}", since.to_rfc3339_opts(SecondsFormat::Secs, true));
    debug!("Using time filter: {}", query);

    let path = format!("repos/{}/issues", repo);
    debug!("Requesting issues from path: {}", path);

    let response = make_github_api_request(&path, Some(&query))?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    trace!("Response body length: {} bytes", text.len());

    debug!("Deserializing JSON response into Issues struct");
    let issues = serde_json::from_str::<Issues>(&text).map_err(|e| {
        warn!("Failed to deserialize issues: {}", e);
        e
    })?;

    debug!("Successfully fetched {} issues from {}", issues.len(), repo);
    trace!("Issues: {:?}", issues);

    Ok(issues)
}

/// Fetches pull requests from a GitHub repository that have been updated within the last `days`.
pub fn get_github_pull_requests(repo: &String, days: u64) -> Result<PullRequests, Box<dyn Error>> {
    debug!(
        "Fetching pull requests from repository {} updated in the last {} days",
        repo, days
    );

    let now = Utc::now();
    let since = now.checked_sub_days(Days::new(days)).unwrap();
    let query = format!("since={}", since.to_rfc3339_opts(SecondsFormat::Secs, true));
    debug!("Using time filter: {}", query);

    let path = format!("repos/{}/pulls", repo);
    debug!("Requesting pull requests from path: {}", path);

    let response = make_github_api_request(&path, Some(&query))?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    trace!("Response body length: {} bytes", text.len());

    debug!("Deserializing JSON response into PullRequests struct");
    let pull_requests = serde_json::from_str::<PullRequests>(&text).map_err(|e| {
        warn!("Failed to deserialize pull requests: {}", e);
        e
    })?;

    debug!(
        "Successfully fetched {} pull requests from {}",
        pull_requests.len(),
        repo
    );
    trace!("Pull requests: {:?}", pull_requests);

    Ok(pull_requests)
}

/// Fetches detailed information for a specific pull request from a GitHub repository.
pub fn get_github_pull_request(repo: &String, pr: u64) -> Result<PullRequest, Box<dyn Error>> {
    debug!(
        "Fetching details for pull request #{} from repository {}",
        pr, repo
    );

    let path = format!("repos/{}/pulls/{}", repo, pr);
    debug!("Requesting PR from path: {}", path);

    let response = make_github_api_request(&path, None)?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    trace!("Response body length: {} bytes", text.len());

    debug!("Deserializing JSON response into PullRequest struct");
    let pull_request = serde_json::from_str::<PullRequest>(&text).map_err(|e| {
        warn!("Failed to deserialize pull request: {}", e);
        e
    })?;

    debug!("Successfully fetched details for PR #{}", pr);
    trace!("Pull request details: {:?}", pull_request);

    Ok(pull_request)
}

/// Fetches the patch content for a specific pull request from a GitHub repository.
pub fn get_github_pull_request_patch(repo: &String, pr: u64) -> Result<String, Box<dyn Error>> {
    debug!(
        "Fetching patch for pull request #{} from repository {}",
        pr, repo
    );

    let path = format!("{}/pull/{}.patch", repo, pr);
    debug!("Requesting patch from path: {}", path);

    let response = make_github_request(&path)?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    debug!(
        "Successfully fetched patch for PR #{} (size: {} bytes)",
        pr,
        text.len()
    );
    trace!(
        "First 100 chars of patch: {}",
        if text.len() > 100 {
            &text[..100]
        } else {
            &text
        }
    );

    Ok(text)
}

/// Fetches detailed information for a specific issue from a GitHub repository.
pub fn get_github_issue(repo: &String, issue: u64) -> Result<Issue, Box<dyn Error>> {
    debug!(
        "Fetching details for issue #{} from repository {}",
        issue, repo
    );

    let path = format!("repos/{}/issues/{}", repo, issue);
    debug!("Requesting issue from path: {}", path);

    let response = make_github_api_request(&path, None)?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    trace!("Response body length: {} bytes", text.len());

    debug!("Deserializing JSON response into Issue struct");
    let issue_data = serde_json::from_str::<Issue>(&text).map_err(|e| {
        warn!("Failed to deserialize issue: {}", e);
        e
    })?;

    debug!("Successfully fetched details for issue #{}", issue);
    trace!("Issue details: {:?}", issue_data);

    Ok(issue_data)
}

/// Fetches all comments for a specific issue from a GitHub repository.
pub fn get_github_issue_comments(repo: &String, issue: u64) -> Result<Comments, Box<dyn Error>> {
    debug!(
        "Fetching comments for issue #{} from repository {}",
        issue, repo
    );

    let path = format!("repos/{}/issues/{}/comments", repo, issue);
    debug!("Requesting comments from path: {}", path);

    let response = make_github_api_request(&path, None)?;
    debug!("Received response, extracting text");

    let text = response.text()?;
    trace!("Response body length: {} bytes", text.len());

    debug!("Deserializing JSON response into Comments struct");
    let comments = serde_json::from_str::<Comments>(&text).map_err(|e| {
        warn!("Failed to deserialize comments: {}", e);
        e
    })?;

    debug!(
        "Successfully fetched {} comments for issue #{}",
        comments.len(),
        issue
    );
    trace!("Comments: {:?}", comments);

    Ok(comments)
}
