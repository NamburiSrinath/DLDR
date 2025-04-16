<b>Blog link:</b> https://semver.org/

---
This is a straight-forward read.
- Version lock - We will be depending on particular version of other packages 
- Version promiscuity - We will make a strong assumption that the future versions of the dependent packages will also work for our usecase.
- X.Y.Z - Major, Minor and Patches
- Bug fixes not affecting the API will increase the Patch (Z), backward compatibility API changes will change the Minor (Y) and backward incompatible API changes will update the major (X)
- If our library works on 3.1.0 for a package, then it's assumed to work for all the minor and patch versions i.e >3.1.0, < 4.0.0 assuming it'that package is following semantic versioning and thus we won't be in dependency hell either locking to a specific version or being too relaxed.