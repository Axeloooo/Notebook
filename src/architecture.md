# Architecture

## Table of Contents

- [Abstraction](#abstraction)

## Abstraction

High-level representation or simplification of complex systems, design or structures.

### Decomposition

Segregation is the idea of breaking down large entities into into smaller and more specialized ones. Create more modular, maintainable, and flexible software designs.

```java
// Class representing a customer
class Cutomer {
  private String name;
  private String email;

  public Customer(String name, String email) {
    this.name = name;
    this.email = email;
  }

  // Getters and setters ...
}

// Class responsible for managing customers
class customerManager {
  public void addCustomer(Customer customer) {
    // Code for adding a new customer
    System.out.println("Customer added successfully.");
  }

  public void updateCustomer(Customer customer) {
    // Code for updating a customer
    System.out.println("Customer information updated.");
  }
}
```

### Coupling

Coupling is the density of dependencies among classes. If a class changes and there is high coupling, many other classes will need to change as well.

- Tight coupling:

  - Customer class creates an Order object directly within its method.
  - Order class maintains a direct reference to the Customer object (customer attribute).

```java
// Class representing a customer
class Cutomer {
  private String name;
  private String email;

  public Customer(String name, String email) {
    this.name = name;
    this.email = email;
  }

  public Order createOrder(String orderDetails) {
    return new Order(this.orderDetails);
  }

  // Getters and setters ...
}

// Class representing an order
class Order {
  private Customer customer;
  private String details;

  public Order(Customer customer, String details) {
    this.customer = customer; // Reference to the customer object
    this.details = details;
  }

  public void displayOrder() {
    System.out.println("Order details: " + this.details);
    System.out.println("Customer name: " + this.customer.getName());
    System.out.println("Customer email: " + this.customer.getEmail());
  }

  // Getters and setters ...
}
```

- Loose coupling:

  - Order class no longer directly references the Customer class.
  - Customer information (name and email) referred as parameters with methods get() and set().

```java
// Class representing a customer
class Cutomer {
  private String name;
  private String email;

  public Customer(String name, String email) {
    this.name = name;
    this.email = email;
  }

  // Getters and setters ...
}

// Class representing an order
class Order {
  private String details;
  private String customerName;
  private String customerEmail;

  public Order(String details, String customerName, String customerEmail) {
    this.details = details;
    this.customerName = customerName;
    this.customerEmail = customerEmail;
  }

  public void displayOrder() {
    System.out.println("Order details: " + this.details);
    System.out.println("Customer name: " + this.customerName);
    System.out.println("Customer email: " + this.customerEmail);
  }

  // Getters and setters ...
}
```

### Cohesion

Cohesion is the degree of unity or closeness among the elements within a class. Each component should represent a single concept. All logic/data of the component should be directly applicable to the concept.

- Low cohesion:

  - Customer mixing responsibilities with createOrder and sendEmail.
  - Order mixing responsabilities with calculateTotal and sendConfirmationEmail.

```java
// Class representing a customer
class Cutomer {
  private String name;
  private String email;
  private int age;

  public Customer(String name, String email, int age) {
    this.name = name;
    this.email = email;
    this.age = age;
  }

  public void createOrder(String product, int quantity) {
    // Create order - Diverse responsibility
  }

  public void sendEmail(String message) {
    // Send email - Unrelated responsibility
  }

  // Getters and setters ...
}

// Class representing an order
class Order {
  private String product;
  private int quantity;
  private double totalAmount;

  public Order(String product, int quantity, double totalAmount) {
    this.product = product;
    this.quantity = quantity;
    this.totalAmount = totalAmount;
  }

  public void calculateTotal() {
    // Calculate total amount - Diverse responsibility
  }

  public void sendConfirmationEmail() {
    // Send confirmation email - Unrelated responsibility
  }

  // Getters and setters ...
}
```

- High cohesion:

  - Order class no longer directly references the Customer class.
  - Customer information (name and email) referred as parameters with methods get() and set().

```java
// Class representing a customer
class Cutomer {
  private String name;
  private String email;

  public Customer(String name, String email) {
    this.name = name;
    this.email = email;
  }

  // Getters and setters ...
}

// Class representing an order
class Order {
  private String details;
  private String customerName;
  private String customerEmail;

  public Order(String details, String customerName, String customerEmail) {
    this.details = details;
    this.customerName = customerName;
    this.customerEmail = customerEmail;
  }

  public void displayOrder() {
    System.out.println("Order details: " + this.details);
    System.out.println("Customer name: " + this.customerName);
    System.out.println("Customer email: " + this.customerEmail);
  }

  // Getters and setters ...
}
```

### Agile-Driven

Architectural approach aligns with the principles and values of agile methodologies. Agile approach emphasizes flexibility, collaboration and responsiveness to change.
