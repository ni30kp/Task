# Development Notes

## Issues encountered during development

### Database setup
- Had to figure out pgvector extension installation on macOS
- Initially tried with SQLite but vector search was too slow
- Settled on PostgreSQL with pgvector - much better performance

### OpenAI API
- Hit rate limits during testing with large batches
- Added fallback to sentence-transformers model (all-MiniLM-L6-v2)
- Works well enough for demo purposes

### Frontend
- Struggled with CORS issues initially
- Had to add proper error handling for when backend is down
- Quote region display was showing "undefined" - fixed by getting value from form

### Performance
- Initial response times were 2-3 seconds
- Optimized by batching embedding generation
- Now consistently under 500ms for most queries

## TODO for production
- [ ] Add proper logging
- [ ] Set up monitoring
- [ ] Add rate limiting
- [ ] Implement caching layer
- [ ] Add user authentication
- [ ] Set up CI/CD pipeline

## Testing notes
- Tested with various French queries - works well
- Quote generation logic seems accurate
- Feedback system stores data but could use more sophisticated learning
- Need to test with larger datasets

## Deployment
- Docker setup works but could be optimized
- Environment variable management could be cleaner
- Database migrations not implemented yet
