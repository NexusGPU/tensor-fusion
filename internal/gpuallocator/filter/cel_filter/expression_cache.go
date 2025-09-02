package cel_filter

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	"github.com/google/cel-go/cel"
)

// CachedCELProgram represents a compiled CEL program with metadata
type CachedCELProgram struct {
	Program     cel.Program
	Expression  string
	CreatedAt   time.Time
	AccessedAt  time.Time
	AccessCount int64
}

// ExpressionCache provides caching for compiled CEL expressions
type ExpressionCache struct {
	cache   map[string]*CachedCELProgram
	mutex   sync.RWMutex
	maxSize int
	maxAge  time.Duration
	env     *cel.Env

	// Metrics
	hits   int64
	misses int64
}

// NewExpressionCache creates a new CEL expression cache
func NewExpressionCache(maxSize int, maxAge time.Duration) (*ExpressionCache, error) {
	env, err := createCELEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create CEL environment: %w", err)
	}

	cache := &ExpressionCache{
		cache:   make(map[string]*CachedCELProgram, maxSize),
		maxSize: maxSize,
		maxAge:  maxAge,
		env:     env,
	}

	// Start cleanup goroutine
	go cache.cleanupExpiredEntries(context.Background())

	return cache, nil
}

// GetOrCompileProgram returns a cached program or compiles and caches a new one
func (c *ExpressionCache) GetOrCompileProgram(expression string) (cel.Program, error) {
	hash := c.hashExpression(expression)

	c.mutex.RLock()
	if cached, exists := c.cache[hash]; exists {
		// Check if entry is still valid
		if time.Since(cached.CreatedAt) < c.maxAge {
			cached.AccessedAt = time.Now()
			cached.AccessCount++
			c.hits++
			c.mutex.RUnlock()
			return cached.Program, nil
		}
	}
	c.mutex.RUnlock()

	// Cache miss or expired - compile new program
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Double-check after acquiring write lock
	if cached, exists := c.cache[hash]; exists && time.Since(cached.CreatedAt) < c.maxAge {
		cached.AccessedAt = time.Now()
		cached.AccessCount++
		c.hits++
		return cached.Program, nil
	}

	// Compile the expression
	ast, issues := c.env.Compile(expression)
	if issues != nil && issues.Err() != nil {
		c.misses++
		return nil, fmt.Errorf("failed to compile CEL expression %q: %w", expression, issues.Err())
	}

	program, err := c.env.Program(ast)
	if err != nil {
		c.misses++
		return nil, fmt.Errorf("failed to create CEL program: %w", err)
	}

	// Check if cache is full and evict least recently used entry
	if len(c.cache) >= c.maxSize {
		c.evictLRU()
	}

	// Cache the compiled program
	c.cache[hash] = &CachedCELProgram{
		Program:     program,
		Expression:  expression,
		CreatedAt:   time.Now(),
		AccessedAt:  time.Now(),
		AccessCount: 1,
	}

	c.misses++
	return program, nil
}

// hashExpression creates a hash for caching expressions
func (c *ExpressionCache) hashExpression(expression string) string {
	hash := sha256.Sum256([]byte(expression))
	return fmt.Sprintf("%x", hash)
}

// evictLRU removes the least recently used entry from cache
func (c *ExpressionCache) evictLRU() {
	var oldestKey string
	var oldestTime time.Time = time.Now()

	for key, cached := range c.cache {
		if cached.AccessedAt.Before(oldestTime) {
			oldestTime = cached.AccessedAt
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(c.cache, oldestKey)
	}
}

// cleanupExpiredEntries removes expired entries periodically
func (c *ExpressionCache) cleanupExpiredEntries(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.mutex.Lock()
			now := time.Now()
			for key, cached := range c.cache {
				if now.Sub(cached.CreatedAt) > c.maxAge {
					delete(c.cache, key)
				}
			}
			c.mutex.Unlock()
		}
	}
}

// GetStats returns cache statistics
func (c *ExpressionCache) GetStats() CacheStats {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	return CacheStats{
		Size:     len(c.cache),
		MaxSize:  c.maxSize,
		Hits:     c.hits,
		Misses:   c.misses,
		HitRatio: float64(c.hits) / float64(c.hits+c.misses),
	}
}

// CacheStats represents cache performance statistics
type CacheStats struct {
	Size     int
	MaxSize  int
	Hits     int64
	Misses   int64
	HitRatio float64
}

// Clear removes all entries from the cache
func (c *ExpressionCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache = make(map[string]*CachedCELProgram, c.maxSize)
	c.hits = 0
	c.misses = 0
}
