# Next Steps After Adding GitHub Secrets

## ✅ Step 1: Verify Secrets Are Added Correctly

### Check Your GitHub Secrets
1. Go to your repository on GitHub
2. Navigate to: **Settings → Secrets and variables → Actions**
3. Verify these secrets exist:

**Required Secrets Checklist:**
- [ ] `AWS_ACCESS_KEY_ID`
- [ ] `AWS_SECRET_ACCESS_KEY`
- [ ] `MLFLOW_TRACKING_URI` (format: `https://dagshub.com/USERNAME/REPO.mlflow`)
- [ ] `DAGSHUB_USERNAME`
- [ ] `DAGSHUB_TOKEN`
- [ ] `ALPHA_VANTAGE_KEY`
- [ ] `DOCKER_HUB_USERNAME`
- [ ] `DOCKER_HUB_TOKEN`

### Verify Secret Values (Optional)
- Make sure `MLFLOW_TRACKING_URI` matches your Dagshub repository
- Ensure `DOCKER_HUB_USERNAME` matches your Docker Hub account
- Verify Docker Hub token has **Read & Write** permissions

---

## ✅ Step 2: Set Up Branch Protection Rules

### For `test` Branch
1. Go to: **Settings → Branches**
2. Click **Add rule** or edit existing rule for `test`
3. Configure:
   - **Branch name pattern**: `test`
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: **1**
   - ✅ **Require status checks to pass before merging**
     - ✅ Check the box: **Require branches to be up to date before merging**
     - In the search box, type: `train-and-compare`
     - Select `train-and-compare` from the dropdown (it appears after workflow runs once)
   - ✅ **Include administrators** (optional but recommended)

**Note**: The status check `train-and-compare` comes from the job name in `.github/workflows/ci-dev-to-test.yml`. If you don't see it in the dropdown:
  1. First create a test PR to trigger the workflow
  2. Wait for workflow to complete
  3. Return to branch protection settings - the check will now appear

### For `main` Branch
1. Click **Add rule** for `main`
2. Configure:
   - **Branch name pattern**: `main`
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: **1**
   - ✅ **Require status checks to pass before merging**
     - ✅ Check the box: **Require branches to be up to date before merging**
     - In the search box, type: `build-and-deploy`
     - Select `build-and-deploy` from the dropdown (it appears after workflow runs once)
   - ✅ **Include administrators**

**Note**: The status check `build-and-deploy` comes from the job name in `.github/workflows/cd-test-to-main.yml`. If you don't see it in the dropdown:
  1. First create a test PR to trigger the workflow
  2. Wait for workflow to complete
  3. Return to branch protection settings - the check will now appear

---

## ✅ Step 3: Test Workflows Step by Step

### Test 1: Feature → Dev Workflow

**Purpose**: Verify linting and unit tests work

1. **Create a feature branch:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/test-ci-workflow
   ```

2. **Make a small change:**
   ```bash
   # Edit any file in src/ or tests/
   # For example, add a comment to src/api.py
   ```

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "test: verify CI workflow"
   git push origin feature/test-ci-workflow
   ```

4. **Create PR to `dev` branch:**
   - Go to GitHub → Pull Requests → New Pull Request
   - Base: `dev`, Compare: `feature/test-ci-workflow`
   - Create PR

5. **Verify workflow runs:**
   - Go to **Actions** tab
   - You should see "CI - Feature to Dev" workflow running
   - Check that it completes successfully
   - Verify:
     - ✅ Linting step passes (or shows warnings)
     - ✅ Unit tests run
     - ✅ Coverage report uploaded

**Expected Result**: Workflow completes (may have warnings, that's OK)

---

### Test 2: Dev → Test Workflow

**Purpose**: Verify model training and CML comparison

1. **Merge the previous PR** (or create a new one from `dev` to `test`)

2. **Create PR from `dev` to `test`:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b test/test-model-training
   # Make a small change
   git commit -m "test: verify model training workflow"
   git push origin test/test-model-training
   ```
   - Create PR: Base `test`, Compare `test/test-model-training`

3. **Verify workflow runs:**
   - Go to **Actions** tab
   - Look for "CI - Dev to Test (Model Training & Comparison)"
   - Check steps:
     - ✅ DVC pull succeeds
     - ✅ Model training completes
     - ✅ CML comparison runs
     - ✅ Report posted as PR comment

4. **Check PR comments:**
   - Go to your PR
   - Look for a comment from CML with model comparison report
   - Should show metrics comparison

**Expected Result**: 
- Model trains successfully
- CML report appears in PR comments
- If no production model exists, it will say "No production model found"

**Note**: This workflow may take 5-10 minutes due to model training

---

### Test 3: Test → Main Workflow

**Purpose**: Verify Docker build and deployment

**Prerequisites**: 
- You need at least one model in MLflow Model Registry (from Test 2)
- Docker Hub repository should exist (or will be created automatically)

1. **Promote a model to Production** (if you have one in Staging):
   - Go to Dagshub → MLflow → Model Registry
   - Find your model in "Staging"
   - Click "Transition to Production"

2. **Create PR from `test` to `main`:**
   ```bash
   git checkout test
   git pull origin test
   git checkout -b main/test-deployment
   # Make a small change
   git commit -m "test: verify CD workflow"
   git push origin main/test-deployment
   ```
   - Create PR: Base `main`, Compare `main/test-deployment`

3. **Verify workflow runs:**
   - Go to **Actions** tab
   - Look for "CD - Test to Main (Deploy to Production)"
   - Check steps:
     - ✅ Model fetched from MLflow
     - ✅ Docker image builds successfully
     - ✅ Image pushed to Docker Hub
     - ✅ Deployment verification passes

4. **Verify Docker Hub:**
   - Go to https://hub.docker.com
   - Check your repository: `YOUR_USERNAME/mlops-stock-prediction-api`
   - Should see:
     - `latest` tag
     - Commit SHA tag
   - Verify image was pushed recently

**Expected Result**: 
- Docker image builds and pushes successfully
- Health check passes
- Image visible on Docker Hub

---

## ✅ Step 4: Verify Everything Works

### Quick Verification Checklist

#### GitHub Actions
- [ ] All three workflows exist in `.github/workflows/`
- [ ] Workflows trigger on correct branch PRs
- [ ] No syntax errors in workflow files

#### Secrets
- [ ] All 8 secrets added to GitHub
- [ ] No typos in secret names
- [ ] Values are correct (especially MLflow URI format)

#### Branch Protection
- [ ] `test` branch requires 1 approval
- [ ] `main` branch requires 1 approval
- [ ] Status checks are required

#### MLflow Integration
- [ ] MLflow tracking URI is correct
- [ ] Dagshub credentials are valid
- [ ] Can access MLflow UI (optional check)

#### Docker Hub
- [ ] Docker Hub token has Read & Write permissions
- [ ] Username matches your Docker Hub account
- [ ] Repository will be created automatically on first push

---

## 🔍 How to Check if Everything is Working

### Method 1: Check Workflow Runs
1. Go to **Actions** tab in GitHub
2. Look for recent workflow runs
3. Green checkmark ✅ = Success
4. Red X ❌ = Failed (check logs)

### Method 2: Check PR Status
1. Open any PR
2. Look at the "Checks" section
3. Should show workflow status
4. Must pass before merging (if branch protection enabled)

### Method 3: Check Docker Hub
1. Visit: `https://hub.docker.com/r/YOUR_USERNAME/mlops-stock-prediction-api`
2. Should see pushed images
3. Check tags: `latest` and commit SHA

### Method 4: Check MLflow
1. Visit your Dagshub MLflow UI
2. Should see:
   - Training runs logged
   - Models in Model Registry
   - Metrics tracked

---

## 🐛 Common Issues and Troubleshooting

### Issue 1: Workflow Not Triggering
**Symptoms**: PR created but no workflow runs

**Solutions**:
- Check workflow file syntax (YAML indentation)
- Verify PR is targeting correct branch (`dev`, `test`, or `main`)
- Check if files changed match `paths` filter in workflow
- Ensure workflow file is in `.github/workflows/` directory

### Issue 2: Secrets Not Found
**Symptoms**: Error like "Secret not found" or authentication fails

**Solutions**:
- Verify secret name matches exactly (case-sensitive)
- Check secret value is correct
- Ensure secrets are repository secrets (not environment secrets)
- Re-add secret if needed

### Issue 3: Docker Build Fails
**Symptoms**: Docker build step fails

**Solutions**:
- Check Dockerfile syntax
- Verify model file exists in `models/` directory
- Check Docker Hub credentials are correct
- Ensure Docker Hub token has Write permission

### Issue 4: MLflow Connection Fails
**Symptoms**: Cannot connect to MLflow or fetch models

**Solutions**:
- Verify `MLFLOW_TRACKING_URI` format: `https://dagshub.com/USERNAME/REPO.mlflow`
- Check Dagshub username and token are correct
- Ensure MLflow tracking is enabled in Dagshub
- Verify repository name matches

### Issue 5: CML Report Not Appearing
**Symptoms**: Model trains but no PR comment

**Solutions**:
- Check `GITHUB_TOKEN` is available (auto-provided)
- Verify CML is installed: `pip install cml`
- Check workflow logs for CML errors
- Ensure PR is not from a fork (CML needs write access)

### Issue 6: Model Registry Empty
**Symptoms**: CD workflow can't find production model

**Solutions**:
- Train a model first (run dev → test workflow)
- Manually promote model to Production in MLflow UI
- Or workflow will use Staging model as fallback

---

## 📊 Expected Workflow Durations

- **Feature → Dev**: 2-5 minutes (linting + tests)
- **Dev → Test**: 5-15 minutes (model training + comparison)
- **Test → Main**: 3-8 minutes (Docker build + push + verify)

---

## ✅ Success Criteria

Everything is working correctly if:

1. ✅ PRs to `dev` trigger linting and tests
2. ✅ PRs to `test` trigger model training and CML comparison
3. ✅ PRs to `main` trigger Docker build and push
4. ✅ CML reports appear in PR comments
5. ✅ Docker images appear on Docker Hub
6. ✅ Models are registered in MLflow Model Registry
7. ✅ Branch protection blocks merges without approval
8. ✅ All workflows complete successfully (green checkmarks)

---

## 🚀 Next Actions

Once everything is verified:

1. **Start using the pipeline:**
   - Create feature branches for new work
   - Follow the branching model: feature → dev → test → main

2. **Monitor workflows:**
   - Check Actions tab regularly
   - Review CML reports for model performance
   - Monitor Docker Hub for new deployments

3. **Promote models:**
   - After successful test → main merge, promote model to Production in MLflow
   - This ensures next deployment uses the best model

4. **Set up alerts** (optional):
   - GitHub email notifications for workflow failures
   - Docker Hub webhooks for deployments

---

## 📝 Quick Test Commands

### Test FastAPI Locally
```bash
# Make sure model exists
ls models/stock_model.pkl

# Run API
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health
```

### Test Docker Build Locally
```bash
# Build image
docker build -f docker/api/Dockerfile -t test-api .

# Run container
docker run -p 8000:8000 test-api

# Test health
curl http://localhost:8000/health
```

### Check Workflow Syntax
```bash
# Install actionlint (optional)
# Or just push and check GitHub Actions tab
```

---

**You're all set!** Start by testing the Feature → Dev workflow, then proceed to the others. If any workflow fails, check the logs in the Actions tab for detailed error messages.

