# Development

This is an example workflow describing the development process for
EasyScience projects.

## 🚀 Step 1: Create GitHub Repositories

### Required repositories

- `project-name` (home repository, always)
- `project-name-lib` (for Python libraries, optional)
- `project-name-app` (for Qt QML desktop applications, optional)

## 🛠️ Step 2: Initialize Projects Using Copier

### Clone repositories

- Clone all project-related repositories:
  ```bash
  git clone https://github.com/easyscience/dynamics.git
  git clone https://github.com/easyscience/dynamics-lib.git
  git clone https://github.com/easyscience/dynamics-app.git
  ```

### Install Pixi

- Install Pixi following the
  [official guide](https://pixi.sh/latest/installation).

### Generate Project Description (Home Repository)

- Navigate to the home repository:
  ```bash
  cd dynamics
  ```
- Set up Pixi and Copier:
  ```bash
  pixi init
  pixi install
  pixi add copier
  ```
- Generate project description:
  ```bash
  pixi run copier copy gh:easyscience/templates .
  ```
- Commit and push:
  ```bash
  git add -A
  git commit -m "Initial project description using Copier"
  git push origin master
  ```
- Navigate back to the parent directory:
  ```bash
  cd ..
  ```

### Apply Templates to Library / Application Repositories

- Navigate to the target repository:
  ```bash
  cd dynamics-lib
  ```
- Set up Pixi and Copier:
  ```bash
  pixi init
  pixi install
  pixi add copier
  ```
- Apply templates (shared and library-specific):
  ```bash
  pixi run copier copy gh:easyscience/templates-shared . --data-file ../dynamics/.project.yaml
  pixi run copier copy gh:easyscience/templates-lib . --data-file ../dynamics/.project.yaml
  ```
- Install development dependencies:
  ```bash
  pixi run post-install
  ```
- Copy logos and branding into docs/:
  ```bash
  pixi run docs-update-assets
  ```
- Update SPDX license headers in all files:
  ```bash
  pixi run spdx-update
  ```
- Format Python and non-Python files:
  ```bash
  pixi run fix
  ```
- Commit and push:
  ```bash
  git add -A
  git commit -m "Initial project setup using Copier templates"
  git push origin master
  ```
- Navigate back to the parent directory:
  ```bash
  cd ..
  ```

## 🛡️ Step 3: Post-Initialization Repository Setup

- Create and publish `develop` branch:
  ```bash
  git checkout -b develop
  git push -u origin develop
  ```
- The `gh-pages` branch is created automatically by GitHub Action
  `docs.yaml` (using mike when deploying docs)
- Set branch protection rules for `master` (default), `develop`, and
  `gh-pages`

## 🔄 Step 4: Updating Existing Repositories

When templates change, repositories must be updated.

- Apply updates from shared templates, if any:
  ```bash
  pixi run copier-update-shared
  ```
- Commit changes
- Apply updates from library templates, if any:
  ```bash
  pixi run copier-update-lib
  ```
- Commit changes

## Step 5: Making and Pushing Changes

### Create a new feature branch

- Checkout/switch to the `develop` branch:
  ```bash
  git checkout develop
  ```
- Create a new branch from `develop`:
  ```bash
  git checkout -b new-feature
  ```
- Push the new branch to the remote repository:
  ```bash
  git push -u origin new-feature
  ```

### Commit and push changes

Make incremental changes in the code, checking code quality and testing.

#### Pre-commit checks

- Check code quality (configuration is in `pyproject.toml` and
  `prettierrc.toml`):
  ```bash
  pixi run pre-commit-check
  ```
- Fix some code quality issues automatically:
  ```bash
  pixi run fix
  ```
- Fix the rest, if any, manually
- Commit changes

### Pre-push checks

- Run tests and checks before pushing changes:
  ```bash
  pixi run pre-push
  ```
- Fix issues, if any, manually
- Push changes

### Individual tests and checks (as needed)

- Run unit tests:
  ```bash
  pixi run unit-tests
  ```
- Run integration tests:
  ```bash
  pixi run integration-tests
  ```
- Test tutorials as Python scripts:
  ```bash
  pixi run script-tests
  ```

### Test and docstring coverage (as needed)

- Check docstring coverage:
  ```bash
  pixi run docstring-coverage
  ```
- Run unit tests with coverage:
  ```bash
  pixi run unit-tests-coverage
  ```
- Run integration tests with coverage:
  ```bash
  pixi run integration-tests-coverage
  ```
- Or just run them all sequentially:
  ```bash
  pixi run cov
  ```

## Building and Checking Documentation with MkDocs (as needed)

- Prepare notebooks:
  ```bash
  pixi run notebook-prepare
  ```
- Build and serve documentation:
  ```bash
  pixi run docs-serve
  ```
- Test the documentation locally (built in the `site/` directory). E.g.,
  on macOS, open the site in the default browser via the terminal:
  ```bash
  open http://127.0.0.1:8000
  ```

## Step 6: Create a Pull Request

When everything is implemented, and once all CI checks are finished and
green at
[dynamics-lib](https://github.com/easyscience/dynamics-lib/actions) or
[dynamics-app](https://github.com/easyscience/dynamics-lib/actions).

- Create a pull request on the
  [dynamics-lib](https://github.com/easyscience/dynamics-lib/pulls) or
  [dynamics-app](https://github.com/easyscience/dynamics-app/pulls) and
  request a review from team members.
- When making a title for the PR, keep in mind that this title will be
  used in autogenerated release notes.
- Add one of the required labels (needed for automatic version bumps and
  categories in autogenerated release notes):
  - `[scope] bug`
  - `[scope] documentation`
  - `[scope] enhancement`
  - `[scope] maintenance`
  - `[scope] significant`
- After approval, merge the pull request into the `develop` branch using
  "Squash and merge" option
- Delete the branch remotely:
  ```bash
  git push origin --delete new-feature
  ```
