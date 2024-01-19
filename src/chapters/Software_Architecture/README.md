# Software Architecture

---

## Table of Contents

- [Abstraction](#abstraction)
  - [Decomposition](#decomposition)
  - [Coupling](#coupling)
  - [Cohesion](#cohesion)
  - [Agile-Driven](#agile-driven)
- [Styles](#styles)
  - [Layered](#layered)
  - [MVC (Model, View, Controller)](#mvc-model-view-controller)
  - [Pipeline](#pipeline)
  - [Microkernel](#microkernel)
  - [Service-Based](#service-based)
  - [Event-Driven](#event-driven)
  - [Other Architectures](#other-architectures)

---

## Abstraction

High-level representation or simplification of complex systems, design or structures.

![Abstraction](./images/image1.png)

### Decomposition

Segregation is the idea of breaking down large entities into into smaller and more specialized ones. Create more modular, maintainable, and flexible software designs.

#### Poor Decomposition

- Customer class is responsible for all entities.
- Customer class is responsible for all operations.

![Poor decomposition](./images/image2.png)

#### Improved Decomposition

- Customer class is responsible for representing customer to access information.
- CustomerManager class is responsible for dealing with customer-related operations.

![Improved decomposition](./images/image3.png)

### Coupling

Coupling is the density of dependencies among classes. If a class changes and there is high coupling, many other classes will need to change as well.

#### Tight Coupling

- Customer class creates an Order object directly within its method.
- Order class maintains a direct reference to the Customer object (customer attribute).

![Tight coupling](./images/image4.png)

#### Loose Coupling

- Order class no longer directly references the Customer class.
- Customer information (name and email) referred as parameters with methods get() and set().

![Loose coupling](./images/image5.png)

### Cohesion

Cohesion is the degree of unity or closeness among the elements within a class. Each component should represent a single concept. All logic/data of the component should be directly applicable to the concept.

#### Low Cohesion

- Customer mixing responsibilities with createOrder and sendEmail.
- Order mixing responsabilities with calculateTotal and sendConfirmationEmail.

![Low cohesion](./images/image6.png)

#### High Cohesion

- Order class no longer directly references the Customer class.
- Customer information (name and email) referred as parameters with methods get() and set().

![High cohesion](./images/image7.png)

### Agile-Driven

Architectural approach aligns with the principles and values of agile methodologies. Agile approach emphasizes flexibility, collaboration and responsiveness to change.

![Agile-Driven](./images/image8.png)

---

## Styles

Recurring patterns used in organizing the structure and components of a software system. Choosing the right architecture style is a critical decision that influences various aspects of software development.

### Layered

Each layer provides a specific functionalities and services to the layers above it.

Key characteristics:

- Simplicity
- Modularity
- Maintainability
- Separation of Concerns.

![Layered](./images/image9.png)

### MVC (Model, View, Controller)

Separate the concerns of data management, user interface and application logic.

Key characteristics:

- Separation of concerns
- User interaction
- Loose coupling
- Layered-Oriented

![MVC](./images/image10.png)

#### Model

Encapsulates the data and provides methods to manipulate and access data.

#### View

Represents the user interface elements and visual components.

#### Controller

Handles user input, processes requests, and updates the Model accordingly.

### Pipeline

Structures the processing of data or tasks as a series of connected processing stages.

Key characteristics:

- Sequential
- Modularity
- Parallelism
- Isolation

![Pipeline](./images/image11.png)

### Microkernel

Structure by separating the core functionality (microkernel) from additional services and functionalities.

Key characteristics:

- Flexibility
- Modularity
- Reduced complexity
- Portability

![Microkernel](./images/image12.png)

### Service-Based

Structures an application as a collection independently deployable services.

Key characteristics:

- Loose coupling
- Interoperability
- Independently deployable
- Reusability

![Service-Based](./images/image13.png)

### Event-Driven

The system flow is determined by events, such as user actions, messages from other systems.

Key characteristics:

- Events
- Responsiveness
- Communication
- Scalability

![Event Driven](./images/image14.png)

### Other Architectures

- Space-Based
- Orchestration-Driven Service-Oriented
- Microservices

![Others](./images/image15.png)

---
