# DevOps

---

## Table of Contents

- [Data Formats](#data-formats)
  - [XML](#xml)
  - [JSON](#json)
  - [YAML](#yaml)
- [Git](#git)
  - [Setting Up Git](#setting-up-git)
  - [Commit](#commit)
  - [Tag](#tag)
  - [Branch](#branch)
  - [Terminology](#terminology)
  - [Distributed Version Control System](#distributed-version-control-system)
  - [Setting Up A Repository](#setting-up-a-repository)
  - [Inspecting A Repository](#inspecting-a-repository)
  - [Undoing Commits And Changes](#undoing-commits-and-changes)
  - [Pull Requests](#pull-requests)
  - [Merge](#merge)
  - [Branch Workflow](#branch-workflow)
- [Virtualization](#virtualization)
  - [Hardware Virtualization](#hardware-virtualization)
  - [Terminology](#terminology)
  - [Advantages](#advantages)
- [Docker](#docker)
  - [Benefits](#benefits)
  - [Containers vs VMs](#containers-vs-vms)
  - [Basic Docker Workflow](#basic-docker-workflow)
  - [Image](#image)
  - [Container](#container)
  - [Image vs Container](#image-vs-container)
  - [Dockerfile](#dockerfile)
  - [Mount Volumes](#mount-volumes)
  - [Publish Port](#publish-port)
  - [Docker Compose](#docker-compose)

---

## Data Formats

We need a standard format to allow a diverse set of software to communicate with each other, and for humans to interface with it. All of the data formats that we’ll discuss in this chapter have broad support over a multitude of languages and operating systems.

- **XML** - machine to machine

- **JSON** - machine to machine

- **YAML** – human to machine

### XML

XML is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable. The design goals of XML emphasize simplicity, generality, and usability across the Internet.

#### Features

- XML files end in `.xml`.

- Root, children nodes, attributes.

- Namespaces (Used to resolve naming conflicts).

![XML](./images/image1.png)

#### Benefits

- Support inter-business transactions.

- Maintain data integrity.

  - Verify data accuracy.
  - Automatically customize data presentation for different users.
  - Store data consistently across multiple platforms.

- Improve search efficiency.

- Design flexible applications.

### JSON

JSON (JavaScript Object Notation) is a lightweight data-interchange format. It is easy for humans to read and write. It is easy for machines to parse and generate.

#### Features

- JSON files end in `.json`.

- The whole thing is wrapped in curly braces {}. This is very common, and it indicates that JSON objects are contained inside.

- Think of “objects” as key-value pairs, or dictionaries.

- JSON objects always use string values when describing the keys.

- JSON list indicated by the square brackets [].

- Data Types:
  - Number
  - String
  - Boolean
  - Array
  - Object
  - Null

![JSON](./images/image2.png)

#### Working with JSON in Python

- JSON enjoys wide support across a myriad of languages
- You will often be able to simply import a JSON data structure into constructs of a given language, simply with a one-line command

```python
import json

item = {
    "name": "egg",
    "price": 0.99,
}

with open('data.json', 'w') as f:
    json.dump(item, f) # write python object to json file

with open('data.json', 'r') as f:
    data = json.load(f) # load json data into python object

print(data) # {'name': 'egg', 'price': 0.99}
```

### YAML

YAML (YAML Ain't Markup Language) is a human-readable data serialization language. It is commonly used for configuration files, automation workflow, or providing a data set
to work with.

#### Features

- YAML files end in `.yaml` or `.yml`.

- YAML is a superset of JSON.

- YAML is a human-readable data serialization language.

- YAML is a strict indentation language.

- YAML is case sensitive.

![YAML](./images/image3.png)

#### Working with YAML in Python

```python
import yaml

item = {
    "name": "egg",
    "price": 0.99,
}

with open('data.yaml', 'w') as f:
    yaml.dump(item, f) # write python object to yaml file

with open('data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader) # load yaml data into python object

print(data) # {'name': 'egg', 'price': 0.99}
```

Note: The yaml module is not part of the standard library, so you’ll need to install it first. You can do so with pip:

```bash
pip install pyyaml
```

---

## Git

![Git](./images/image10.png)

Source code management systems are essential for any software development project. They allow you to keep track of your software at the source level. You can track changes, revert to previous stages, and work on different branches at the same time. Code is organized within a repository.

### Setting Up Git

```bash
git config --global user.name <name>    # Set user name

git config --global user.email <email>  # Set user email
```

### Commit

- Specific snapshot within the development tree.

- Collection of changes applied to a project's files.

- Metadata about the change.

- Identified by a unique `SHA-1 Hash`.

```bash
git commit -m "Add new feature"   # Commit changes

git show 5b8e4f                   # Show commit details
```

### Tag

- Represents a single commit.

- Often human-friendly.

- Version number.

```bash
git tag v1.0.0    # Create a tag

git tag           # List all tags

git show v1.0.0   # Show tag details
```

### Branch

- A history of successive changes to code.

- A new branch may be created at any time, from any existing commit.

- May represent versions of code.

```bash
git branch                        # List all branches

git branch feature/add-btn        # Create a new branch

git checkout feature/add-btn      # Switch to a branch

git checkout -b feature/add-btn   # Create and switch to a branch
```

![Git Branch](./images/image4.png)

### Terminology

- **Working Files**: Files that are currently on your file system.

- **Staging Area**: Files that are ready to be committed. Only files that have been staged will be committed.

- **Checkout**: Replace the current directory files with those from a specific branch or commit.

![Git Model](./images/image5.png)

### Distributed Version Control System

- **Clone**: Creates a full copy of repository on your local machine.

- **Fetch**: Update your local repository with changes from non-local repositories (i.e. GitHub).

- **Fork**: Create a copy of a repository in a remote location (i.e. GitHub).

```bash
git clone https://github.com/username/repo.git   # Clone a repository

git fetch origin                                # Fetch changes from remote repository
```

### Setting Up A Repository

- Create a new repository from cero on the command line

```bash
git init                                                    # Initialize a new repository

git remote add origin https://github.com/username/repo.git  # Add a remote repository
```

- Clone an existing repository

```bash
git clone https://github.com/username/repo.git    # Clone a repository

cd repo                                           # Change directory to the repository
```

### Inspecting A Repository

- **git diff**: Show changes between commits, commit and working tree, etc.

- **git status**: Display the state of the working directory and the staging area.

- **git blame**: Examine the history of a file and get context as to who made changes and when.

- **git log**: Show commit logs.

```bash
git diff ./file.txt    # Show changes between commits, commit and working tree, etc.

git status            # Display the state of the working directory and the staging area

git blame ./file.txt  # Examine the history of a file

git log               # Show commit logs
```

![Git diff](./images/image6.png)

### Undoing Commits And Changes

- **git revert**: Create a new commit that undoes the changes from a previous commit.

- **git commit --amend**: Change the last commit.

- **git reset --soft**: Only resets the HEAD to the commit you specified.

- **git reset --mixed**: Resets the HEAD to the commit you specified and resets the staging area.

- **git reset --hard**: Resets the HEAD to the commit you specified and resets the staging area and working directory.

```bash
git revert 5b8e4f    # Create a new commit that undoes the changes from a previous commit

git commit --amend   # Change the last commit

git reset --soft 5b8e4f    # Only resets the HEAD to the commit you specified

git reset --mixed 5b8e4f   # Resets the HEAD to the commit you specified and resets the staging area

git reset --hard 5b8e4f    # Resets the HEAD to the commit you specified and resets the staging area and working directory
```

### Pull Requests

Pull requests are a mechanism for a developer to notify team members that they have completed a feature. Once their feature branch is ready, the developer files a pull request via their SCM (Source Control Management) system. This allows team members to review the changes and discuss any potential modifications.

![Pull Request](./images/image7.png)

### Merge

A merge is the act of integrating the changes from one branch or commit into a second branch. The result is a new commit that is a combination of the two. If a merge fails, git will notify you with a merge conflict. Merge conflixts must be fixed manually, and then added and committed.

**NOTE**: Merge conflicts occur when two branches have changed the same part of the same file, and then those branches are merged together. Git will not be able to automatically determine what the correct content should be or if both changes should be included, so you must inspect and resolve the conflict manually.

```bash
git merge feature/add-btn    # Merge a branch into the current branch
```

### Branch Workflow

Feature branching is a git workflow that allows multiple developers to work on a particular feature in isolation. Each feature is developed in a dedicated branch, and once it is ready, it is merged into the main branch.

![Branch Workflow](./images/image9.png)

---

## Virtualization

![Virtualization](./images/image8.png)

### Hardware Virtualization

- Abstract underlying physical hardware from operating systems and applications.

- Allows multiple guest operating systems to run in parallel.

- Physical resources are shared among all guest OS and virtualization software.

### Terminology

- **Host OS/Host Machine**: Physical hardware/server.

- **Hypervisor**: The virtualization software, acts as the true OS for the server.

- **Guest OS/Virtual Machines**: Instances of the virtualized OS, running in emulated environments. Guest thinks it’s running on real hardware.

### Advantages

- **Sandbox**: Isolated from the host OS.

- **Networked**: Access over the network.

- **Portable**: Run on any host OS.

- **Snapshots**: Saved and restored to previous states.

---

## Docker

![Docker](./images/image11.png)

Docker is a platform for developers and sysadmins to develop, deploy, and run applications with containers. The use of Linux containers to deploy applications is called containerization.

### Benefits

- **Flexible**: Even the most complex applications can be containerized.

- **Lightweight**: Containers leverage and share the host kernel.

- **Interchangeable**: You can deploy updates and upgrades on-the-fly.

- **Portable**: You can build locally, deploy to the cloud, and run anywhere.

- **Scalable**: You can increase and automatically distribute container replicas.

- **Stackable**: You can stack services vertically and on-the-fly.

### Containers vs VMs

Container are isolated, but share OS and, where appropriate, bins/libraries. VMs are isolated, but each has its own OS, bins, and libraries. This results in significantly faster deployment, much less overhead, easier migration and faster restart.

![Containers vs VMs](./images/image12.png)

### Basic Docker Workflow

![Workflow](./images/image13.png)

### Image

Persisted snapshot that can be run as a container.

- **images**: List all local images.

- **run**: Create a container from an image and execute a command in it.

- **tag**: Tag an image.

- **pull**: Download an image from a registry.

- **rmi**: Delete a local image.

```bash
docker images                     # List all local images

docker run -it ubuntu /bin/bash   # Create a container from an image and execute a command in it

docker tag ubuntu my-ubuntu       # Tag an image

docker pull ubuntu                # Download an image from a registry

docker rmi ubuntu                 # Delete a local image
```

### Container

Runnable instance of an image.

- **ps**: List all running containers.

- **ps -a**: List all containers (running and stopped).

- **top**: Display processes of a container.

- **start**: Start a stopped container.

- **stop**: Stop a running container.

- **pause**: Pause all processes within a container.

- **rm**: Delete a container.

- **commit**: Create an image from a container.

```bash
docker ps                      # List all running containers

docker ps -a                   # List all containers (running and stopped)

docker top <container_id>      # Display processes of a container

docker start <container_id>    # Start a stopped container

docker stop <container_id>     # Stop a running container

docker pause <container_id>    # Pause all processes within a container

docker rm <container_id>       # Delete a container
```

### Image vs Container

![Image vs Container](./images/image14.png)

### Dockerfile

Dockerfile is a text document that contains all the commands a user could call on the command line to assemble an image. Can be versioned in a version control system like Git or
SVN, along with all dependencies. Docker Hub can automatically build images based on
dockerfiles on Github WorkFlows.

- Configuration is done in a `Dockerfile`.

```Dockerfile
FROM ubuntu
ENV DOCK_MESSAGE Hello My World
ADD dir /files
CMD ["bash", "echo $DOCK_MESSAGE"]
```

The above `Dockerfile` will create an image from the `ubuntu` image, set an environment variable, add a directory to the image, and run a command.

- Run the following commands to build and run the image:

```bash
docker build .              # Build an image from a Dockerfile

docker inspect <image_id>   # Return low-level information on Docker objects
```

### Mount Volumes

Volumes allow data to persist beyond the lifetime of a container.

```bash
docker run -it -v /hostLog:/log ubuntu  # Mount a volume to a container
```

Run a second container, volume can be shared between containers.

```bash
docker run -it --volumes-from <container_id> ubuntu  # Mount a volume from another container
```

### Publish Port

Port mapping allows a container to expose a port to the host.

```bash
docker run -p 8080:80 nginx  # Publish a container's port to the host
```

This will allow you to access the container's port 80 from the host's port 8080.

- `8080` Host Port (computer running the container)

- `80` Container Port (inside the container)

You can link containers together, so that they can communicate with each other.

```bash
docker run --link <container_id>:<alias> <image>  # Link a container to another container
```

### Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services. Then, with a single command, you create and start all the services from your configuration.

- Configuration is done in a `docker-compose.yml` file.

```yaml
version: "3"

services:
  postgres:
    image: postgres
    environment:
      POSTGRES_PASSWORD: root
    ports:
      - "5432:5432"
    volumes:
      - db:/var/lib/postgresql/data
    networks:
      - postgres

  pgadmin4:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: pgadmin4@pgadmin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    networks:
      - postgres
    volumes:
      - pgadmin:/var/lib/pgadmin

networks:
  postgres:
    driver: bridge

volumes:
  db:
  pgadmin:
```

The above `docker-compose.yml` file will create two services, `postgres` and `pgadmin4`. The `postgres` service will use the `postgres` image, and the `pgadmin4` service will use the `dpage/pgadmin4` image. The volume is used to persist the data of each service. The network is used to connect both services with each other.

- Run the following commands to start and stop the services:

```bash
docker-compose up    # Create and start all the services from your configuration

docker-compose down  # Stop and remove containers, networks, images, and volumes
```

---
